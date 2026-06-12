use std::time::Duration;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ServeOptions {
    pub allow_remote_shutdown: bool,
    pub idle_timeout: Option<Duration>,
    pub drain_timeout: Option<Duration>,
    pub close_timeout: Option<Duration>,
}

impl From<ServeOptions> for rlmesh_grpc::ServeOptions {
    fn from(value: ServeOptions) -> Self {
        Self {
            allow_remote_shutdown: value.allow_remote_shutdown,
            idle_timeout: value.idle_timeout,
            drain_timeout: value.drain_timeout,
            close_timeout: value.close_timeout,
            token: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_options_default_disables_remote_shutdown() {
        assert_eq!(
            ServeOptions::default(),
            ServeOptions {
                allow_remote_shutdown: false,
                idle_timeout: None,
                drain_timeout: None,
                close_timeout: None,
            }
        );
    }

    #[test]
    fn facade_serve_options_convert_to_transport_options() {
        let options = ServeOptions {
            allow_remote_shutdown: true,
            idle_timeout: Some(Duration::from_secs(1)),
            drain_timeout: Some(Duration::from_secs(2)),
            close_timeout: Some(Duration::from_secs(3)),
        };
        let grpc_options = rlmesh_grpc::ServeOptions::from(options);
        assert_eq!(
            grpc_options.allow_remote_shutdown,
            options.allow_remote_shutdown
        );
        assert_eq!(grpc_options.idle_timeout, options.idle_timeout);
        assert_eq!(grpc_options.drain_timeout, options.drain_timeout);
        assert_eq!(grpc_options.close_timeout, options.close_timeout);
    }
}
