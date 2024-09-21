use std::{f32::consts::PI, fs::File, io::Read};

use glam::{Vec3 as FVec3, Vec3Swizzles};
use itertools::Itertools;
use luisa::lang::types::vector::{Mat3, Vec2, Vec3};
use sefirot::prelude::*;
use sefirot_testbed::{App, KeyCode};

#[tracked]
fn falloff(x: Expr<f32>) -> Expr<f32> {
    1.0 - (1.0 + x).recip()
}

#[tracked]
fn color(value: Expr<f32>) -> Expr<Vec3<f32>> {
    Vec3::expr(
        falloff(value / 2.0) * value,
        value * 0.3,
        falloff(value * 0.7) + value * 0.3,
    )
}

#[tracked]
fn color_of(value: Expr<u32>, max_value: Expr<f32>) -> Expr<Vec3<f32>> {
    let value = value.cast_f32();
    let value = value / max_value;
    let value = (value + 0.1).ln() - 0.1_f32.ln();
    // let value = (value / 5.0).clamp(0.0, 1.0);
    color(value) * 0.1
}

fn main() {
    let display_size = 2048;
    let sidebar_size = 256;

    let app = App::new("Hilbertdust", [display_size + sidebar_size, display_size])
        .dpi_override(2.0)
        .agx()
        .init();
    let data = File::open(std::env::args().nth(1).unwrap()).unwrap();
    let data = std::io::BufReader::new(data);
    let data = data.bytes().map(|x| x.unwrap()).collect_vec();
    let data_buffer = DEVICE.create_buffer_from_slice(&data);

    let histo_buffer = DEVICE.create_buffer::<u32>(256 * 256 * 256);

    let update_histo_kernel =
        DEVICE.create_kernel_async::<fn(Buffer<u8>, u32)>(&track!(|data, stride| {
            let index = dispatch_id().x * stride;
            let a = data.read(index).cast_u32();
            let b = data.read(index + 1).cast_u32();
            let c = data.read(index + 2).cast_u32();
            histo_buffer
                .atomic_ref(a + b * 256 + c * 65536)
                .fetch_add(1);
        }));

    let texture = DEVICE.create_tex3d::<u32>(PixelStorage::Int1, 256, 256, 256, 1);

    let copy_texture_kernel = DEVICE.create_kernel_async::<fn()>(&track!(|| {
        let id = dispatch_id();
        let index = id.x + id.y * 256 + id.z * 65536;
        texture.write(id, histo_buffer.read(index));
        histo_buffer.write(index, 0);
    }));

    let display_sidebar_kernel =
        DEVICE.create_kernel_async::<fn(u32, u32)>(&track!(|stride, vert_stride| {
            let index = dispatch_id().x * stride + dispatch_id().y * sidebar_size * vert_stride;
            let color = if index < data_buffer.len() as u32 {
                let value = data_buffer.read(index).cast_f32() / 255.0;
                value * Vec3::new(0.2, 1.0, 0.2)
            } else {
                Vec3::expr(1.0, 0.0, 0.0)
            };
            app.display()
                .write(dispatch_id().xy() + Vec2::new(display_size, 0), color);
        }));

    let trace_kernel = DEVICE.create_kernel_async::<fn(f32, Vec3<f32>, Mat3)>(&track!(
        |color_scale, ray_start, view| {
            let ray_start = ray_start * 128.0 + 128.0;
            let ray_dir = view
                * (dispatch_id().xy().cast_f32() - app.display().size().cast_f32() * 0.5)
                    .extend(1.0);
            let ray_dir = ray_dir.normalize();

            let t0 = (0.01 - ray_start) / ray_dir;
            let t1 = (255.99 - ray_start) / ray_dir;
            let tmin = luisa::min(t0, t1).reduce_max();
            let tmax = luisa::max(t0, t1).reduce_min();
            if tmin > tmax {
                // if tmin - tmax < 2.0 {
                //     app.display().write(dispatch_id().xy(), Vec3::splat(1.0));
                // }
                return;
            }

            let ray_start = ray_start + luisa::max(tmin, 0.0) * ray_dir;

            let pos = ray_start.floor().cast_i32();
            let pos = pos.var();

            // Have to transform by grid_scale since it could be uneven?
            let delta_dist = (ray_dir.length() / (ray_dir + f32::EPSILON)).abs();

            let ray_step = ray_dir.signum().cast_i32();
            let side_dist =
                (ray_dir.signum() * (pos.cast_f32() - ray_start) + ray_dir.signum() * 0.5 + 0.5)
                    * delta_dist;
            let side_dist = side_dist.var();

            let color = Vec3::splat(0.0_f32).var();

            for _i in 0_u32.expr()..1000_u32.expr() {
                let mask = side_dist <= luisa::min(side_dist.yzx(), side_dist.zxy());

                *side_dist += mask.select(delta_dist, Vec3::splat_expr(0.0));
                *pos += mask.select(ray_step, Vec3::splat_expr(0_i32));

                let next_t = side_dist.reduce_min();

                if next_t > tmax - tmin {
                    break;
                }

                let value = texture.read(pos.cast_u32());

                *color += color_of(value, color_scale);
            }

            app.display().write(dispatch_id().xy(), color);
        }
    ));

    let mut max_value = 1000.0;
    // *histo
    //     .select_nth_unstable((histo.len() as f32 - 1.0) as usize)
    //     .1 as f32;
    let fov = 0.9_f32;

    let mut horiz_angle = PI / 4.0;
    let mut vert_angle = PI / 4.0;
    let mut scale = 4.0;
    let mut auto_rotate = true;
    let mut update_display = true;
    let mut sidebar_stride = 1;
    let mut sidebar_vert_stride = 1;

    app.run(|rt, scope| {
        if rt.pressed_key(KeyCode::KeyR) {
            auto_rotate = !auto_rotate;
        }
        if auto_rotate {
            horiz_angle += 0.007;
        }

        if rt.pressed_key(KeyCode::KeyA) {
            horiz_angle += 0.01;
            auto_rotate = false;
        }
        if rt.pressed_key(KeyCode::KeyD) {
            horiz_angle -= 0.01;
            auto_rotate = false;
        }
        if rt.pressed_key(KeyCode::KeyW) {
            vert_angle -= 0.01;
            vert_angle = vert_angle.clamp(-PI / 2.0 + 0.001, PI / 2.0 + 0.001);
        }
        if rt.pressed_key(KeyCode::KeyS) {
            vert_angle += 0.01;
            vert_angle = vert_angle.clamp(-PI / 2.0 + 0.001, PI / 2.0 + 0.001);
        }
        if rt.pressed_key(KeyCode::KeyQ) {
            scale *= 1.01;
        }
        if rt.pressed_key(KeyCode::KeyE) {
            scale *= 0.99;
        }
        if rt.pressed_key(KeyCode::KeyZ) {
            max_value *= 1.1;
        }
        if rt.pressed_key(KeyCode::KeyX) {
            max_value *= 0.9;
        }
        if rt.pressed_key(KeyCode::BracketLeft) {
            sidebar_vert_stride = (sidebar_vert_stride - 1).max(1);
        }
        if rt.pressed_key(KeyCode::BracketRight) {
            sidebar_vert_stride += 1;
        }

        let start = FVec3::new(
            horiz_angle.sin() * vert_angle.cos(),
            horiz_angle.cos() * vert_angle.cos(),
            vert_angle.sin(),
        ) * scale;
        let forward = -start.normalize();
        let right = forward.xy().perp().extend(0.0).normalize();
        let down = right.cross(forward).normalize();
        let ratio = (fov / 2.0).tan() / (display_size as f32 / 2.0);

        let start = Vec3::from(start);
        let view: Mat3 = glam::Mat3::from_cols(right * ratio, down * ratio, forward).into();

        if update_display {
            let data_view = 0..data_buffer.len();
            let stride = 1;
            scope.submit([
                update_histo_kernel.dispatch_async(
                    [(data_view.len() - 3) as u32 / stride, 1, 1],
                    &data_buffer.view(data_view),
                    &stride,
                ),
                copy_texture_kernel.dispatch_async([256, 256, 256]),
            ]);
            update_display = false;
        }

        scope.submit([
            display_sidebar_kernel.dispatch_async(
                [sidebar_size, display_size, 1],
                &sidebar_stride,
                &sidebar_vert_stride,
            ),
            trace_kernel.dispatch_async([display_size, display_size, 1], &max_value, &start, &view),
        ]);
    })
}
