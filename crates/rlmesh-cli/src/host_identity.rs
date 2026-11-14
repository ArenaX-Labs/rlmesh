//! `rlmesh host-identity`: the keys a launcher starts a session host under
//! in another process (a container), and the address it dials it by, for
//! launchers that are not written in Rust.

use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlmesh_session::address::{
    HOST_IDENTITY_VAR, identity_key, launch_identity, launched_address, write_private,
};
use serde_json::json;
use zeroize::Zeroizing;

use crate::cli::{
    HostIdentityArgs, HostIdentityCommand, HostIdentityKeyArgs, HostIdentityNewArgs,
    LaunchAddressArgs,
};

pub fn host_identity(args: &HostIdentityArgs, stdout: &mut impl Write) -> Result<()> {
    match &args.command {
        HostIdentityCommand::New(args) => new(args, stdout),
        HostIdentityCommand::Key(args) => key(args, stdout),
    }
}

fn new(args: &HostIdentityNewArgs, stdout: &mut impl Write) -> Result<()> {
    let (identity, key) = launch_identity()?;
    let address = keyed_address(&identity, &args.address)?;
    if let Some(out) = &args.out {
        write_private(out, identity.as_bytes())
            .with_context(|| format!("writing the identity to {}", out.display()))?;
    }
    let printed = args.out.is_none().then_some(identity.as_str());
    if args.json {
        let document = Zeroizing::new(
            json!({
                "identity": printed,
                "identityFile": args.out.as_ref().map(|out| out.display().to_string()),
                "key": key.to_string(),
                "address": address.as_deref(),
            })
            .to_string(),
        );
        writeln!(stdout, "{}", document.as_str())?;
        return Ok(());
    }
    if let Some(identity) = printed {
        writeln!(stdout, "identity={identity}")?;
    }
    write_key(stdout, &key.to_string(), address.as_deref())
}

fn key(args: &HostIdentityKeyArgs, stdout: &mut impl Write) -> Result<()> {
    let identity = match &args.identity_file {
        Some(file) => read_identity(file)?,
        None => Zeroizing::new(std::env::var(HOST_IDENTITY_VAR).map_err(|_| {
            anyhow::anyhow!("set {HOST_IDENTITY_VAR}, or pass --identity-file FILE")
        })?),
    };
    let key = identity_key(&identity)?.to_string();
    let address = keyed_address(&identity, &args.address)?;
    if args.json {
        writeln!(stdout, "{}", json!({"key": key, "address": address}))?;
        return Ok(());
    }
    write_key(stdout, &key, address.as_deref())
}

fn read_identity(file: &Path) -> Result<Zeroizing<String>> {
    let mut text = Zeroizing::new(String::new());
    if file == Path::new("-") {
        std::io::stdin()
            .read_to_string(&mut text)
            .context("reading the identity from stdin")?;
    } else {
        text = Zeroizing::new(
            std::fs::read_to_string(file)
                .with_context(|| format!("reading the identity from {}", file.display()))?,
        );
    }
    if text.trim().is_empty() {
        bail!("{} holds no identity", file.display());
    }
    Ok(text)
}

/// The keyed address `args` asks for, if any.
fn keyed_address(identity: &str, args: &LaunchAddressArgs) -> Result<Option<String>> {
    let Some(session) = &args.address else {
        return Ok(None);
    };
    let mut address = launched_address(identity, session)?;
    address.participant.clone_from(&args.participant);
    // Parsed back so a bad participant name is refused here, not at dial.
    let invite = address.invite();
    match &args.participant {
        Some(_) => rlmesh_session::address::Address::participant(&invite)?,
        None => rlmesh_session::address::Address::session(&invite)?,
    };
    Ok(Some(invite))
}

fn write_key(stdout: &mut impl Write, key: &str, address: Option<&str>) -> Result<()> {
    writeln!(stdout, "key={key}")?;
    if let Some(address) = address {
        writeln!(stdout, "address={address}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn address(address: Option<&str>, participant: Option<&str>) -> LaunchAddressArgs {
        LaunchAddressArgs {
            address: address.map(str::to_string),
            participant: participant.map(str::to_string),
        }
    }

    fn lines(out: &[u8]) -> std::collections::HashMap<String, String> {
        String::from_utf8_lossy(out)
            .lines()
            .filter_map(|line| line.split_once('='))
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    #[test]
    fn new_prints_an_identity_whose_key_reads_back() {
        let mut out = Vec::new();
        let args = HostIdentityNewArgs {
            out: None,
            address: address(Some("127.0.0.1:7600"), Some("env")),
            json: false,
        };
        new(&args, &mut out).unwrap();
        let printed = lines(&out);
        let identity = &printed["identity"];
        let key = &printed["key"];
        assert_eq!(&identity_key(identity).unwrap().to_string(), key);
        assert_eq!(printed["address"], format!("127.0.0.1:7600#{key}/env"));

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("identity");
        std::fs::write(&file, format!("{identity}\n")).unwrap();
        let mut out = Vec::new();
        let args = HostIdentityKeyArgs {
            identity_file: Some(file),
            address: address(Some("127.0.0.1:7600/bench"), None),
            json: true,
        };
        super::key(&args, &mut out).unwrap();
        let document: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(document["key"], key.as_str());
        assert_eq!(document["address"], format!("127.0.0.1:7600/bench#{key}"));
    }

    #[cfg(unix)]
    #[test]
    fn new_out_writes_the_identity_privately_and_prints_only_the_key() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("keys").join("identity");
        let mut out = Vec::new();
        let args = HostIdentityNewArgs {
            out: Some(file.clone()),
            address: address(None, None),
            json: true,
        };
        new(&args, &mut out).unwrap();
        let document: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert!(document["identity"].is_null());
        assert!(document["address"].is_null());
        assert_eq!(document["identityFile"], file.display().to_string());
        let mode = std::fs::metadata(&file).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        let identity = std::fs::read_to_string(&file).unwrap();
        assert!(!String::from_utf8_lossy(&out).contains(identity.trim()));
        assert_eq!(
            identity_key(&identity).unwrap().to_string(),
            document["key"].as_str().unwrap()
        );
    }

    #[test]
    fn a_bad_identity_is_refused_without_repeating_it() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("identity");
        std::fs::write(&file, "hunter2.hunter2.hunter2").unwrap();
        let args = HostIdentityKeyArgs {
            identity_file: Some(file),
            address: address(None, None),
            json: false,
        };
        let error = format!("{:#}", super::key(&args, &mut Vec::new()).unwrap_err());
        assert!(!error.contains("hunter2"), "{error}");
    }
}
