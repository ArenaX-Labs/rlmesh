//! The host's datagram path for a robot's realtime parts: what the handshake
//! offers and the routes a device view lists ([`crate::datagram`] frames the
//! datagrams themselves).

#[allow(unused_imports)]
use crate::prelude::*;

use crate::wire::ids::ChannelId;

/// The host relays robots' realtime parts over UDP (`HandshakeAccept`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DatagramOffer {
    /// Its UDP port, on the host the participant reached.
    pub port: u16,
    /// The datagram header's version ([`crate::datagram::VERSION`]).
    pub version: u8,
}

/// One robot's routes on the datagram path (`DeviceView.datagram`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DatagramRoutes {
    /// The route its probes name.
    pub probe: u16,
    /// Every route of its parts, down and up.
    pub routes: Vec<DatagramRoute>,
}

impl DatagramRoutes {
    /// The route carrying the chunks of commandable part `channel` down.
    pub fn down(&self, channel: ChannelId) -> Option<u16> {
        self.find(channel, false)
    }

    /// The route carrying stream `channel` (samples or `_applied`) up.
    pub fn up(&self, channel: ChannelId) -> Option<u16> {
        self.find(channel, true)
    }

    fn find(&self, channel: ChannelId, up: bool) -> Option<u16> {
        self.routes
            .iter()
            .find(|r| r.channel == channel && r.up == up)
            .map(|r| r.route)
    }
}

/// One route: a datagram names it instead of its key expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DatagramRoute {
    /// Its id.
    pub route: u16,
    /// Down: the commandable part whose chunks it carries. Up: the stream
    /// it carries.
    pub channel: ChannelId,
    /// Board to host.
    pub up: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_channel_has_one_route_each_way() {
        let route = |route, channel, up| DatagramRoute {
            route,
            channel: ChannelId(channel),
            up,
        };
        let routes = DatagramRoutes {
            probe: 1,
            routes: vec![route(2, 5, false), route(3, 5, true), route(4, 6, true)],
        };
        assert_eq!(
            (routes.down(ChannelId(5)), routes.up(ChannelId(5))),
            (Some(2), Some(3))
        );
        assert_eq!(
            (routes.down(ChannelId(6)), routes.up(ChannelId(6))),
            (None, Some(4))
        );
        assert_eq!(routes.up(ChannelId(7)), None);
    }
}
