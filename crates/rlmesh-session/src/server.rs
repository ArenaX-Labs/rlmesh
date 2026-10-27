//! A multi-session host: one [`SessionHost`] per session of a tenant,
//! started when its first participant appears and stopped when its last
//! one leaves.
//!
//! The server watches every participant's liveliness token in its tenant
//! (`{tenant}/s/*/p/*`). A participant declares its token before it waits
//! for its session's host, so the server sees it, starts that session's
//! host on the same transport, and the participant then finds the host as
//! it would a host started ahead of it. The server's queryable
//! (`{tenant}/server`) tells a participant that a host here will serve its
//! session, so one that finds only other sessions' hosts fails at once.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::Mutex;
use tracing::{info, warn};
use zenoh::Wait;
use zenoh::sample::SampleKind;

use crate::error::SessionError;
use crate::host::{HostConfig, SessionHost, token_declared};
use crate::transport::Transport;
use crate::wire::SessionId;
use crate::wire::keys::{session_of, session_server, tenant_participants};

type Hosts = Arc<Mutex<HashMap<SessionId, SessionHost>>>;

/// A participant's liveliness token appeared (`true`) or left.
type Presence = (bool, SessionId, String);

/// A running multi-session host. Dropping it stops its sessions; prefer
/// [`SessionServer::shutdown`], which waits for them.
pub struct SessionServer {
    hosts: Hosts,
    _server: zenoh::query::Queryable<()>,
    _watch: zenoh::pubsub::Subscriber<()>,
}

/// Removes `key` from `members` when its token left: not when its
/// participant rejoined and its new token is declared ahead of the old one's
/// late delete, however many other members the session has.
async fn member_left(members: &mut HashSet<String>, key: &str, control: &zenoh::Session) -> bool {
    if members.contains(key)
        && let Ok(expr) = zenoh::key_expr::KeyExpr::try_from(key)
        && token_declared(control, &expr).await
    {
        return false;
    }
    members.remove(key);
    true
}

impl SessionServer {
    /// Serves every session of `config.tenant` on `transport`, each with
    /// `config` (its `session` replaced), starting `config.session` at once.
    pub async fn start(transport: Transport, config: HostConfig) -> Result<Self, SessionError> {
        let hosts: Hosts = Arc::default();
        hosts.lock().await.insert(
            config.session,
            SessionHost::start(transport.clone(), config.clone()).await?,
        );
        let control = transport.control().clone();
        let tenant_id = config.tenant.clone();
        let tenant = config.tenant.to_string();
        // Presence changes are applied in order, by one task.
        let (presence, mut changes) = tokio::sync::mpsc::unbounded_channel::<Presence>();
        tokio::spawn({
            let hosts = hosts.clone();
            let control = control.clone();
            async move {
                let mut present: HashMap<SessionId, HashSet<String>> = HashMap::new();
                while let Some((joined, session, key)) = changes.recv().await {
                    let members = present.entry(session).or_default();
                    if joined {
                        members.insert(key);
                    } else if !member_left(members, &key, &control).await {
                        continue;
                    }
                    let empty = members.is_empty();
                    if empty {
                        present.remove(&session);
                    }
                    let mut hosts = hosts.lock().await;
                    if !empty && !hosts.contains_key(&session) {
                        let config = HostConfig {
                            session,
                            ..config.clone()
                        };
                        match SessionHost::start(transport.clone(), config).await {
                            Ok(host) => {
                                info!(%session, "session started for its first participant");
                                hosts.insert(session, host);
                            }
                            Err(err) => warn!(%session, %err, "session did not start"),
                        }
                    } else if empty
                        && !joined
                        && let Some(host) = hosts.remove(&session)
                    {
                        info!(%session, "session stopped: its last participant left");
                        host.shutdown().await;
                    }
                }
            }
        });
        let watch = control
            .liveliness()
            .declare_subscriber(tenant_participants(&tenant_id))
            .history(true)
            .callback(move |sample| {
                let key = sample.key_expr().as_str();
                let Some((key_tenant, session)) = session_of(key) else {
                    return;
                };
                if key_tenant != tenant {
                    return;
                }
                let joined = sample.kind() == SampleKind::Put;
                let _ = presence.send((joined, session, key.to_string()));
            })
            .await?;
        let server = control
            .declare_queryable(session_server(&tenant_id))
            .callback(|query| {
                let key = query.key_expr().clone();
                let _ = query.reply(key, Vec::<u8>::new()).wait();
            })
            .await?;
        Ok(Self {
            hosts,
            _server: server,
            _watch: watch,
        })
    }

    /// The sessions it hosts.
    pub async fn sessions(&self) -> Vec<SessionId> {
        self.hosts.lock().await.keys().copied().collect()
    }

    /// Stops every session (participants see their hosts leave).
    pub async fn shutdown(self) {
        let hosts: Vec<SessionHost> = self.hosts.lock().await.drain().map(|(_, h)| h).collect();
        for host in hosts {
            host.shutdown().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};

    use super::*;
    use crate::transport::open_local_listeners;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_late_delete_of_a_rejoined_members_old_token_keeps_it_beside_others() {
        let (transport, _) = open_local_listeners(IpAddr::from(Ipv4Addr::LOCALHOST))
            .await
            .expect("transport");
        let control = transport.control();
        let (a, b) = ("rlmesh/test/members/a", "rlmesh/test/members/b");
        // A rejoined: its new token is declared when its old one's delete lands.
        let token = control.liveliness().declare_token(a).await.expect("token");
        let mut members: HashSet<String> = [a.to_string(), b.to_string()].into();
        assert!(!member_left(&mut members, a, control).await);
        assert!(members.contains(a), "a rejoined member stays tracked");
        // B left for good: no token of its own.
        assert!(member_left(&mut members, b, control).await);
        assert_eq!(members, HashSet::from([a.to_string()]));
        // Then A leaves for good: its replacement token's delete removes it.
        token.undeclare().await.expect("undeclare");
        assert!(member_left(&mut members, a, control).await);
        assert!(members.is_empty(), "{members:?}");
        transport.close().await.expect("close");
    }
}
