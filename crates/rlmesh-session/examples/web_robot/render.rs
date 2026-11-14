//! A synthetic 640x480 camera: a side view of the arm that moves with the
//! joints, a frame counter, the session and edge modes and the lease holder,
//! plus sensor-like noise so the JPEG lands in the 40-60 KB range of a real
//! camera frame.

pub const W: usize = 640;
pub const H: usize = 480;

pub struct Scene<'a> {
    pub q: &'a [f64; 7],
    pub gripper: f64,
    pub frame: u64,
    pub mode: &'a str,
    pub edge: &'a str,
    pub holder: &'a str,
    pub t_ms: f64,
}

pub struct Renderer {
    rgb: Vec<u8>,
    noise: Vec<i8>,
    rng: u64,
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer {
    pub fn new() -> Self {
        let mut r = Self {
            rgb: vec![0; W * H * 3],
            noise: vec![0; W * H * 3 + 4096],
            rng: 0x9E37_79B9_7F4A_7C15,
        };
        for i in 0..r.noise.len() {
            r.noise[i] = ((r.next() >> 59) as i8) - 16; // [-16, 15]
        }
        r
    }

    fn next(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    pub fn render(&mut self, s: &Scene) -> &[u8] {
        // Background: vertical gradient, floor, grid.
        for y in 0..H {
            let (r, g, b) = if y < 400 {
                let k = y as f64 / 400.0;
                (
                    (22.0 + 20.0 * k) as u8,
                    (28.0 + 26.0 * k) as u8,
                    (40.0 + 34.0 * k) as u8,
                )
            } else {
                (58, 62, 66)
            };
            let row = &mut self.rgb[y * W * 3..(y + 1) * W * 3];
            for x in 0..W {
                let grid =
                    (y >= 400 && (x + (y - 400) * 2) % 64 < 2) || (y >= 400 && (y - 400) % 16 == 0);
                let d = if grid { 18 } else { 0 };
                row[x * 3] = r + d;
                row[x * 3 + 1] = g + d;
                row[x * 3 + 2] = b + d;
            }
        }
        // Table.
        self.rect(380, 330, 600, 345, [120, 92, 64]);
        self.rect(395, 345, 405, 400, [90, 70, 50]);
        self.rect(575, 345, 585, 400, [90, 70, 50]);
        // A cube on the table that slowly rotates colour.
        let hue = (s.t_ms / 40.0) as u32 % 255;
        self.rect(470, 300, 500, 330, [200, hue as u8, 80]);

        // Arm: planar chain from joints 1..=5, base yaw (joint 0) foreshortens the chain.
        let base = (220.0f64, 400.0f64);
        let lens = [120.0, 100.0, 80.0, 45.0, 30.0];
        let yaw = s.q[0].cos().abs().max(0.25);
        let mut ang = -std::f64::consts::FRAC_PI_2;
        let mut p = base;
        self.disk(base.0, base.1, 26.0, [70, 74, 82]);
        let colors = [
            [230, 120, 40],
            [240, 150, 50],
            [245, 175, 60],
            [200, 200, 210],
            [180, 180, 190],
        ];
        for (i, l) in lens.iter().enumerate() {
            ang += s.q[i + 1];
            let n = (p.0 + l * ang.cos() * yaw, p.1 + l * ang.sin());
            let w = 16.0 - i as f64 * 2.5;
            self.capsule(p, n, w, colors[i]);
            self.disk(p.0, p.1, w * 0.8, [40, 44, 52]);
            p = n;
        }
        // Gripper fingers, opened by the gripper (0..1).
        let open = 6.0 + 12.0 * s.gripper.clamp(0.0, 1.0);
        let (dx, dy) = (-(ang.sin()), ang.cos());
        for sgn in [-1.0, 1.0] {
            let a = (p.0 + dx * open * sgn, p.1 + dy * open * sgn);
            let b = (a.0 + 18.0 * ang.cos(), a.1 + 18.0 * ang.sin());
            self.capsule(a, b, 4.0, [210, 210, 220]);
        }

        // Border: the arm's edge mode.
        let mc = match s.edge {
            "driving" => [40, 200, 90],
            "hold" => [240, 180, 30],
            "idle" => [120, 130, 140],
            _ => [230, 50, 50],
        };
        self.rect(0, 0, W, 6, mc);
        self.rect(0, H - 6, W, H, mc);
        self.rect(0, 0, 6, H, mc);
        self.rect(W - 6, 0, W, H, mc);

        // Text.
        self.text(18, 18, 3, &format!("FRAME {:06}", s.frame), [235, 235, 235]);
        self.text(18, 48, 2, &format!("MODE {} EDGE {}", s.mode, s.edge), mc);
        self.text(18, 66, 2, &format!("HOLDER {}", s.holder), [200, 200, 200]);
        let qs: Vec<String> = s.q.iter().map(|v| format!("{v:+.2}")).collect();
        self.text(18, 452, 2, &format!("Q {}", qs.join(" ")), [180, 190, 200]);

        // Sensor noise (shifted per frame so it is not static and JPEG cannot fold it away).
        let off = (self.next() % 4096) as usize;
        for (i, px) in self.rgb.iter_mut().enumerate() {
            *px = px.saturating_add_signed(self.noise[i + off] / 2);
        }
        &self.rgb
    }

    fn rect(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, c: [u8; 3]) {
        for y in y0.min(H)..y1.min(H) {
            for x in x0.min(W)..x1.min(W) {
                let i = (y * W + x) * 3;
                self.rgb[i..i + 3].copy_from_slice(&c);
            }
        }
    }

    fn disk(&mut self, cx: f64, cy: f64, r: f64, c: [u8; 3]) {
        let (x0, x1) = ((cx - r).max(0.0) as usize, ((cx + r) as usize).min(W - 1));
        let (y0, y1) = ((cy - r).max(0.0) as usize, ((cy + r) as usize).min(H - 1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (dx, dy) = (x as f64 - cx, y as f64 - cy);
                if dx * dx + dy * dy <= r * r {
                    let i = (y * W + x) * 3;
                    self.rgb[i..i + 3].copy_from_slice(&c);
                }
            }
        }
    }

    fn capsule(&mut self, a: (f64, f64), b: (f64, f64), r: f64, c: [u8; 3]) {
        let (x0, x1) = (
            (a.0.min(b.0) - r).max(0.0) as usize,
            ((a.0.max(b.0) + r) as usize).min(W - 1),
        );
        let (y0, y1) = (
            (a.1.min(b.1) - r).max(0.0) as usize,
            ((a.1.max(b.1) + r) as usize).min(H - 1),
        );
        let (vx, vy) = (b.0 - a.0, b.1 - a.1);
        let ll = (vx * vx + vy * vy).max(1e-9);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (px, py) = (x as f64 - a.0, y as f64 - a.1);
                let t = ((px * vx + py * vy) / ll).clamp(0.0, 1.0);
                let (dx, dy) = (px - t * vx, py - t * vy);
                if dx * dx + dy * dy <= r * r {
                    let shade = 1.0 - 0.25 * ((dx * dx + dy * dy).sqrt() / r);
                    let i = (y * W + x) * 3;
                    for (px, ch) in self.rgb[i..i + 3].iter_mut().zip(c) {
                        *px = (f64::from(ch) * shade) as u8;
                    }
                }
            }
        }
    }

    fn text(&mut self, x: usize, y: usize, scale: usize, s: &str, c: [u8; 3]) {
        for (ci, ch) in s.chars().enumerate() {
            let g = glyph(ch.to_ascii_uppercase());
            for (row, bits) in g.iter().enumerate() {
                for col in 0..5 {
                    if bits & (0x10 >> col) != 0 {
                        let px = x + ci * 6 * scale + col * scale;
                        let py = y + row * scale;
                        self.rect(px, py, px + scale, py + scale, c);
                    }
                }
            }
        }
    }
}

fn glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '+' => [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
        '/' => [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00],
        _ => [0; 7],
    }
}
