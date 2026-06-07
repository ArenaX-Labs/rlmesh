use std::fmt;
use std::path::PathBuf;

use rlmesh_grpc::helpers::{BindTarget, parse_bind_target, parse_env_connect_target};

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectAddress {
    Tcp(String),
    Unix(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindAddress {
    Tcp { host: String, port: u16 },
    Unix { path: PathBuf },
}

impl ConnectAddress {
    pub fn parse(value: impl AsRef<str>) -> Result<Self> {
        let target = parse_env_connect_target(value.as_ref()).map_err(Error::from)?;
        Ok(match target.unix_path() {
            Some(path) => Self::Unix(path.clone()),
            None => Self::Tcp(target.display_address().to_string()),
        })
    }

    pub fn as_str(&self) -> String {
        match self {
            Self::Tcp(value) => value.clone(),
            Self::Unix(path) => format!("unix://{}", path.display()),
        }
    }
}

impl fmt::Display for ConnectAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

impl BindAddress {
    pub fn parse(value: impl AsRef<str>) -> Result<Self> {
        Ok(
            match parse_bind_target(value.as_ref()).map_err(Error::from)? {
                BindTarget::Tcp { host, port } => Self::Tcp { host, port },
                BindTarget::Unix { path } => Self::Unix { path },
            },
        )
    }

    pub fn display_address(&self) -> String {
        match self {
            Self::Tcp { host, port } => format!("tcp://{host}:{port}"),
            Self::Unix { path } => format!("unix://{}", path.display()),
        }
    }
}

impl fmt::Display for BindAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.display_address().fmt(f)
    }
}
