//! The OS calls the CLI makes through libc (Unix): a board's serial port,
//! opened as tether spec §5.1 step 2 and §14.2 say, and the machine's host
//! name. Neither std nor the workspace's crates set termios or the modem
//! lines.
#![allow(unsafe_code)] // libc: termios, modem-line ioctls, poll, gethostname.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::time::Duration;

use crate::provision::Port;

/// Espressif's USB vendor id: its native USB (USB-Serial-JTAG, 303a:1001,
/// and TinyUSB CDC) keeps quiet until DTR is asserted and resets on RTS.
const ESPRESSIF: &str = "303a";

/// A serial port at 115200 baud, raw, held exclusively.
pub struct Tty {
    file: File,
}

fn check(result: libc::c_int) -> io::Result<()> {
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// A modem-line ioctl's result: a port with no modem lines (a pty, as
/// socat gives a test board) refuses them with `ENOTTY` or `EINVAL`, and
/// has nothing to set.
fn modem_lines(result: libc::c_int) -> io::Result<()> {
    match check(result) {
        Err(err) if matches!(err.raw_os_error(), Some(libc::ENOTTY | libc::EINVAL)) => Ok(()),
        other => other,
    }
}

/// The USB vendor id behind `path` (`/dev/ttyUSB0`), from sysfs (Linux).
fn usb_vendor(path: &Path) -> Option<String> {
    let name = std::fs::canonicalize(path).ok()?;
    let name = name.file_name()?.to_str()?;
    let mut dir = std::fs::canonicalize(format!("/sys/class/tty/{name}/device")).ok()?;
    for _ in 0..4 {
        if let Ok(vendor) = std::fs::read_to_string(dir.join("idVendor")) {
            return Some(vendor.trim().to_ascii_lowercase());
        }
        dir = dir.parent()?.to_path_buf();
    }
    None
}

impl Tty {
    /// Opens `path` exclusively at 115200 baud, never 1200 (the bootloader
    /// touch), with the modem lines §5.1 gives its adapter: RTS low
    /// always, DTR high only for Espressif's native USB. A USB-UART bridge
    /// (CP210x, CH340, CH9102) or an unknown adapter gets both low, since
    /// asserting either resets the board behind it. Stale input is
    /// flushed.
    // TODO: the vendor id comes from Linux's sysfs only; on macOS every
    // port is treated as a USB-UART bridge (DTR low), so a native-USB
    // ESP32-S3 there needs IOKit's vendor id when one is provisioned on a Mac.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOCTTY | libc::O_NONBLOCK)
            .open(path)?;
        file.try_lock().map_err(|_| {
            io::Error::new(
                io::ErrorKind::ResourceBusy,
                format!("{} is open in another program", path.display()),
            )
        })?;
        let fd = file.as_raw_fd();
        // The lock keeps out other rlmesh processes; TIOCEXCL keeps out
        // every other open of the tty (an IDE's serial monitor), until
        // this one ends (tether spec §7).
        // SAFETY: `fd` is open for the life of `file`.
        check(unsafe { libc::ioctl(fd, libc::TIOCEXCL as _) }).map_err(|err| {
            io::Error::new(
                io::ErrorKind::ResourceBusy,
                format!("{}: exclusive open: {err}", path.display()),
            )
        })?;
        let native = usb_vendor(path).as_deref() == Some(ESPRESSIF);
        // SAFETY: `fd` is open for the life of `file`; every pointer is to
        // a live local of the type the call takes.
        unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFL);
            check(flags)?;
            check(libc::fcntl(fd, libc::F_SETFL, flags & !libc::O_NONBLOCK))?;
            let mut termios: libc::termios = std::mem::zeroed();
            check(libc::tcgetattr(fd, &mut termios))?;
            libc::cfmakeraw(&mut termios);
            check(libc::cfsetspeed(&mut termios, libc::B115200))?;
            termios.c_cflag |= libc::CLOCAL | libc::CREAD;
            termios.c_cflag &= !libc::HUPCL;
            check(libc::tcsetattr(fd, libc::TCSANOW, &termios))?;
            let low: libc::c_int = if native {
                libc::TIOCM_RTS
            } else {
                libc::TIOCM_RTS | libc::TIOCM_DTR
            };
            modem_lines(libc::ioctl(fd, libc::TIOCMBIC as _, &low))?;
            if native {
                let high: libc::c_int = libc::TIOCM_DTR;
                modem_lines(libc::ioctl(fd, libc::TIOCMBIS as _, &high))?;
            }
            check(libc::tcflush(fd, libc::TCIOFLUSH))?;
        }
        Ok(Self { file })
    }
}

impl Drop for Tty {
    fn drop(&mut self) {
        // A pty keeps its flags past this close: open it to others again.
        // SAFETY: `file` is still open.
        unsafe { libc::ioctl(self.file.as_raw_fd(), libc::TIOCNXCL as _) };
    }
}

impl Port for Tty {
    fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.file.write_all(bytes)?;
        self.file.flush()
    }

    fn recv(&mut self, buf: &mut [u8], wait: Duration) -> io::Result<usize> {
        let mut poll = libc::pollfd {
            fd: self.file.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let ms = libc::c_int::try_from(wait.as_millis()).unwrap_or(libc::c_int::MAX);
        // SAFETY: one live `pollfd`, and its count.
        let ready = unsafe { libc::poll(&mut poll, 1, ms) };
        match ready {
            0 => Ok(0),
            n if n < 0 => {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::Interrupted {
                    Ok(0)
                } else {
                    Err(err)
                }
            }
            _ => self.file.read(buf),
        }
    }
}

/// This machine's host name, without its domain.
pub fn hostname() -> Option<String> {
    let mut buf = [0_u8; 256];
    // SAFETY: `buf` is writable for its whole length.
    check(unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) }).ok()?;
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let name = std::str::from_utf8(&buf[..end]).ok()?;
    let name = name.split('.').next().unwrap_or_default();
    (!name.is_empty()).then(|| name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_port_held_by_provisioning_refuses_every_other_open_until_closed() {
        // SAFETY: the pty's master is opened, granted and unlocked, and
        // its name copied before anything else runs on this thread.
        let (master, slave) = unsafe {
            let master = libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY);
            assert!(master >= 0, "{}", io::Error::last_os_error());
            assert_eq!(libc::grantpt(master), 0);
            assert_eq!(libc::unlockpt(master), 0);
            let name = std::ffi::CStr::from_ptr(libc::ptsname(master));
            (
                master,
                std::path::PathBuf::from(name.to_str().expect("utf-8")),
            )
        };
        let other = || OpenOptions::new().read(true).write(true).open(&slave);
        let tty = Tty::open(&slave).expect("opened");
        // An open that takes no lock, as a serial monitor's.
        let refused = other().expect_err("held exclusively");
        assert_eq!(refused.raw_os_error(), Some(libc::EBUSY), "{refused}");
        drop(tty);
        other().expect("free once closed");
        // SAFETY: the master is still open.
        unsafe { libc::close(master) };
    }
}
