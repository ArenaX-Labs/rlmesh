//! A multi-session host: one [`SessionHost`] per session of a tenant,
//! started when its first participant appears.
//!
//! The server watches every participant's liveliness token in its tenant
//! (`{tenant}/s/*/p/*`). A participant declares its token before it waits
//! for its session's host, so the server sees it, starts that session's
//! host on the same transport, and the participant then finds the host as
//! it would a host started ahead of it. The server's queryable
//! (`{tenant}/server`) tells a participant that a host here will serve its
//! session, so one that finds only other sessions' hosts fails at once.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;
use tracing::{info, warn};
use zenoh::Wait;
use zenoh::sample::SampleKind;

use crate::error::SessionError;
use crate::host::{HostConfig, SessionHost};
use crate::transport::Transport;
use crate::wire::SessionId;
use crate::wire::keys::{session_of, session_server, tenant_participants};

type Hosts = Arc<Mutex<HashMap<SessionId, SessionHost>>>;

/// A running multi-session host. Dropping it stops its sessions; prefer
/// [`SessionServer::shutdown`], which waits for them.
pub struct SessionServer {
    hosts: Hosts,
    _server: zenoh::query::Queryable<()>,
    _watch: zenoh::pubsub::Subscriber<()>,
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
        let watch = control
            .liveliness()
            .declare_subscriber(tenant_participants(&tenant_id))
            .history(true)
            .callback({
                let hosts = hosts.clone();
                move |sample| {
                    if sample.kind() != SampleKind::Put {
                        return;
                    }
                    let Some((key_tenant, session)) = session_of(sample.key_expr().as_str()) else {
                        return;
                    };
                    if key_tenant != tenant {
                        return;
                    }
                    let (hosts, transport) = (hosts.clone(), transport.clone());
                    let config = HostConfig {
                        session,
                        ..config.clone()
                    };
                    tokio::spawn(async move {
                        let mut hosts = hosts.lock().await;
                        if hosts.contains_key(&session) {
                            return;
                        }
                        match SessionHost::start(transport, config).await {
                            Ok(host) => {
                                info!(%session, "session started for its first participant");
                                hosts.insert(session, host);
                            }
                            Err(err) => warn!(%session, %err, "session did not start"),
                        }
                    });
                }
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
