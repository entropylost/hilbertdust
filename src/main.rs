use std::{f32::consts::PI, fs::File, io::Read};

use glam::{Vec3 as FVec3, Vec3Swizzles};
use itertools::Itertools;
use luisa::lang::types::vector::{Mat3, Vec2, Vec3};
use sefirot::prelude::*;
use sefirot_testbed::{App, KeyCode, MouseButton};

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

    let app = App::new(
        "Hilbertdust",
        [display_size + 2 * sidebar_size + 64, display_size],
    )
    .dpi_override(2.0)
    .agx()
    .init();
    let data = File::open(std::env::args().nth(1).unwrap()).unwrap();
    let data = std::io::BufReader::new(data);
    let mut data = data.bytes().map(|x| x.unwrap()).collect_vec();
    let data_len = data.len();
    data.extend(std::iter::repeat(0).take(256));
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
    let update_histo_slices_kernel = DEVICE.create_kernel_async::<fn(Buffer<u8>, u32, u32)>(
        &track!(|data, stride, time_stride| {
            let index = dispatch_id().x * stride + dispatch_id().y * time_stride;
            let a = data.read(index).cast_u32();
            let b = data.read(index + 1).cast_u32();
            histo_buffer
                .atomic_ref(a + b * 256 + dispatch_id().y * 65536)
                .fetch_add(1);
        }),
    );

    let texture = DEVICE.create_tex3d::<u32>(PixelStorage::Int1, 256, 256, 256, 1);

    let draw_square_kernel =
        DEVICE.create_kernel_async::<fn(Vec2<u32>, Vec3<f32>)>(&track!(|offset, color| {
            let pos = dispatch_id().xy() + offset;
            app.display().write(pos, color);
        }));

    let copy_texture_kernel = DEVICE.create_kernel_async::<fn()>(&track!(|| {
        let id = dispatch_id();
        let index = id.x + id.y * 256 + id.z * 65536;
        texture.write(id, histo_buffer.read(index));
        histo_buffer.write(index, 0);
    }));

    let display_sidebar_kernel =
        DEVICE.create_kernel_async::<fn(u32, u32, i32, i32, u32, u32)>(&track!(
            |stride, vert_stride, view_start, view_end, start, section| {
                #[tracked]
                fn nearto(x: Expr<i32>) -> Expr<bool> {
                    (dispatch_id().y.cast_i32() - x).abs() < 5
                }

                let index = start + dispatch_id().x * stride + dispatch_id().y * vert_stride;
                let color = if nearto(view_start / vert_stride.cast_i32())
                    || nearto(view_end / vert_stride.cast_i32())
                {
                    Vec3::expr(0.0, 0.0, 5.0)
                } else if index < data_len as u32 {
                    let value = data_buffer.read(index).cast_f32() / 255.0;
                    value * Vec3::new(0.2, 1.0, 0.2)
                } else {
                    Vec3::expr(1.0, 0.0, 0.0)
                };
                app.display().write(
                    dispatch_id().xy() + Vec2::expr(display_size + sidebar_size * section, 0),
                    color,
                );
            }
        ));

    let trace_kernel = DEVICE.create_kernel_async::<fn(f32, Vec3<f32>, Mat3)>(&track!(
        |color_scale, ray_start, view| {
            let ray_start = ray_start * 128.0 + 128.0;
            let ray_dir =
                view * (dispatch_id().xy().cast_f32() - display_size as f32 * 0.5).extend(1.0);
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
    let fov = 0.9_f32;

    let mut horiz_angle = PI / 4.0;
    let mut vert_angle = PI / 4.0;
    let mut scale = 4.0;
    let mut auto_rotate = true;
    let mut update_display = true;
    let sidebar_stride = 1;
    let second_view_vert_stride =
        sidebar_size * ((data_len as f32 * 1.05) as u32 / sidebar_size / display_size);
    let mut sidebar_vert_stride =
        ((data_len as u32 / sidebar_size / display_size) * sidebar_size).max(sidebar_size);
    let mut second_view = 0..data_len;
    let mut data_view = 0..data_len;
    let mut seeking = false;
    let mut data_stride = 1;
    let mut use_slices = false;

    app.run(|rt, scope| {
        if rt.just_pressed_key(KeyCode::KeyR) {
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
        if rt.just_pressed_key(KeyCode::Period) {
            data_stride += 1;
            if data_stride > 4 {
                data_stride = 1;
            }
            update_display = true;
        }
        if rt.just_pressed_key(KeyCode::Space) {
            use_slices = !use_slices;
            update_display = true;
        }
        if rt.pressed_button(MouseButton::Left) {
            let pos = rt.cursor_position;
            if pos.x > display_size as f32 + sidebar_size as f32 {
                let index = pos.y as usize * second_view_vert_stride as usize;
                if index < data_len {
                    second_view.start = index;
                    second_view.end = second_view
                        .end
                        .max(second_view.start + display_size as usize * sidebar_size as usize);
                    sidebar_vert_stride =
                        ((second_view.len() as u32 / sidebar_size / display_size) * sidebar_size)
                            .max(sidebar_size);
                }
            } else if pos.x > display_size as f32 {
                let index = pos.y as usize * sidebar_vert_stride as usize + second_view.start;
                if index < data_len {
                    data_view.start = index;
                    data_view.end = data_view.end.max(data_view.start + sidebar_size as usize);
                    update_display = true;
                }
            }
        }
        if rt.pressed_button(MouseButton::Right) {
            let pos = rt.cursor_position;
            if pos.x > display_size as f32 + sidebar_size as f32 {
                let index = pos.y as usize * second_view_vert_stride as usize;
                if index < data_len {
                    second_view.end = index;
                    second_view.start = second_view.start.min(
                        second_view
                            .end
                            .saturating_sub(display_size as usize * sidebar_size as usize),
                    );
                    sidebar_vert_stride =
                        ((second_view.len() as u32 / sidebar_size / display_size) * sidebar_size)
                            .max(sidebar_size);
                }
            } else if pos.x > display_size as f32 {
                let index = pos.y as usize * sidebar_vert_stride as usize + second_view.start;
                if index < data_len {
                    data_view.end = index;
                    data_view.start = data_view
                        .start
                        .min(data_view.end.saturating_sub(sidebar_size as usize));
                    update_display = true;
                }
            }
        }
        if rt.just_pressed_key(KeyCode::Backslash) {
            seeking = !seeking;
        }
        if seeking {
            data_view.start += sidebar_vert_stride as usize;
            data_view.end += sidebar_vert_stride as usize;
            update_display = true;
            data_view.end = data_view.end.min(data_len);
            data_view.start = data_view
                .start
                .min(data_view.end.saturating_sub(sidebar_size as usize));
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
            if use_slices {
                let time_stride = data_view.len() as u32 / 256;
                scope.submit([
                    update_histo_slices_kernel.dispatch_async(
                        [data_view.len() as u32 / data_stride / 256, 256, 1],
                        &data_buffer.view(data_view.start..),
                        &data_stride,
                        &time_stride,
                    ),
                    copy_texture_kernel.dispatch_async([256, 256, 256]),
                ]);
            } else {
                scope.submit([
                    update_histo_kernel.dispatch_async(
                        [data_view.len() as u32 / data_stride, 1, 1],
                        &data_buffer.view(data_view.start..),
                        &data_stride,
                    ),
                    copy_texture_kernel.dispatch_async([256, 256, 256]),
                ]);
            }
            update_display = false;
        }

        scope.submit((0..data_stride).map(|i| {
            draw_square_kernel.dispatch_async(
                [48, 48, 1],
                &Vec2::new(display_size + 2 * sidebar_size + 8, 8 + i * 56),
                &Vec3::splat(1.0),
            )
        }));
        if use_slices {
            scope.submit([draw_square_kernel.dispatch_async(
                [48, 48, 1],
                &Vec2::new(display_size + 2 * sidebar_size + 8, display_size - 48 - 8),
                &Vec3::new(1.0, 0.0, 0.0),
            )]);
        }

        scope.submit([
            display_sidebar_kernel.dispatch_async(
                [sidebar_size, display_size, 1],
                &sidebar_stride,
                &sidebar_vert_stride,
                &(data_view.start as i32 - second_view.start as i32),
                &(data_view.end as i32 - second_view.start as i32),
                &(second_view.start as u32),
                &0,
            ),
            display_sidebar_kernel.dispatch_async(
                [sidebar_size, display_size, 1],
                &sidebar_stride,
                &second_view_vert_stride,
                &(second_view.start as i32),
                &(second_view.end as i32),
                &0,
                &1,
            ),
            trace_kernel.dispatch_async([display_size, display_size, 1], &max_value, &start, &view),
        ]);
    })
}
