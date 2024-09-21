use std::{f32::consts::PI, fs::File, io::Read};

use bytemuck::cast_slice_mut;
use glam::{Vec3 as FVec3, Vec3Swizzles};
use itertools::Itertools;
use luisa::lang::types::vector::{Mat3, Vec3};
use sefirot::prelude::*;
use sefirot_testbed::{App, KeyCode};

#[tracked]
fn color(value: Expr<f32>) -> Expr<Vec3<f32>> {
    Vec3::expr(
        0.5 - (value * 2.0).cos() / 2.0,
        value.sin() * 0.3,
        value.sin(),
    )
}

#[tracked]
fn color_of(value: Expr<f32>, max_value: Expr<f32>) -> Expr<Vec3<f32>> {
    let value = value * u16::MAX as f32;
    let value = value / max_value;
    let value = (value + 0.1).ln() - 0.1_f32.ln();
    let value = (value / 5.0).clamp(0.0, 1.0);
    color(value) * 0.1
}

fn main() {
    let app = App::new("Cantor", [2048; 2]).dpi_override(2.0).agx().init();
    let data = File::open(std::env::args().nth(1).unwrap()).unwrap();
    let data = std::io::BufReader::new(data);
    let mut histo = data.bytes().map(|x| x.unwrap()).collect_vec();
    let histo = cast_slice_mut::<u8, u16>(&mut histo);
    let texture = DEVICE.create_tex3d::<f32>(PixelStorage::Short1, 256, 256, 256, 1);
    texture.view(0).copy_from(histo);

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

    let max_value = 1000.0;
    // *histo
    //     .select_nth_unstable((histo.len() as f32 - 1.0) as usize)
    //     .1 as f32;
    let fov = 0.9_f32;

    let mut horiz_angle = PI / 4.0;
    let mut vert_angle = PI / 4.0;
    let mut scale = 2.5;
    let mut auto_rotate = true;

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

        let start = FVec3::new(
            horiz_angle.sin() * vert_angle.cos(),
            horiz_angle.cos() * vert_angle.cos(),
            vert_angle.sin(),
        ) * scale;
        let forward = -start.normalize();
        let right = forward.xy().perp().extend(0.0).normalize();
        let down = right.cross(forward).normalize();
        let ratio = (fov / 2.0).tan() / (rt.display().height() as f32 / 2.0);

        let start = Vec3::from(start);
        let view: Mat3 = glam::Mat3::from_cols(right * ratio, down * ratio, forward).into();

        scope.submit([trace_kernel.dispatch_async(
            [rt.display().width(), rt.display().height(), 1],
            &max_value,
            &start,
            &view,
        )]);
    })
}
