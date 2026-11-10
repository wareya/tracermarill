use std::sync::Arc;

use pixels::{Error, Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::KeyCode;
use winit::window::Window;
use winit_input_helper::WinitInputHelper;

use glam::*;

const WIDTH: u32 = 256;

struct World
{
    field : Box<[f32; WIDTH as usize*WIDTH as usize]>,
    rngstate : u64,
    ncount : u64,
    t_prev : std::time::Instant,
    dt_mem : f64,
}

fn rng_u64(rngstate : &mut u64) -> u64
{
    *rngstate = rngstate.wrapping_add(314159265);
    *rngstate ^= 31415926;
    *rngstate = rngstate.wrapping_mul(3141592653589793);
    
    let mut ret = *rngstate;
    ret ^= ret >> 16;
    ret = ret.wrapping_add(314159265);
    ret ^= ret >> 23;
    return ret;
}
fn rng_f32(rngstate : &mut u64) -> f32
{
    let ret = rng_u64(rngstate) >> (64-24);
    let ret = (ret as f32) / 16777216.0f32;
    return ret;
}

impl World {
    fn new() -> Self
    {
        Self {
            field: Box::new([0.0; WIDTH as usize*WIDTH as usize]),
            rngstate: 9816291,
            ncount: 0,
            t_prev: std::time::Instant::now(),
            dt_mem: 0.01,
        }
    }
    fn continue_trace(&mut self)
    {
        use rayon::prelude::*;
        use std::f32::consts::PI;
        let chunkn = 12;
        let count = (WIDTH*WIDTH) as u64;
        let mut rngstate = self.rngstate + 1;
        let mut mv = vec!(Box::new([0.0; WIDTH as usize*WIDTH as usize]); chunkn);
        mv.par_chunks_mut(1).enumerate().for_each(|(idx, data)|
        {
            let mut rngstate = rngstate + idx as u64;
            
            let la = rng_f32(&mut rngstate) * PI * 2.0;
            let lz = rng_f32(&mut rngstate) * 2.0 - 1.0;
            let lx = (1.0-lz*lz).sqrt() * la.cos();
            let ly = (1.0-lz*lz).sqrt() * la.sin();
            // white furnace
            //let lightdir = vec3(lx, ly, lz).normalize();
            
            // side
            let lightdir = vec3(1.0, 0.0, 0.0).normalize();
            
            let data = &mut data[0];
            
            let schlick_approx = |mut x : f32| { x = 1.0 - x; let x2 = x*x; let x4 = x2*x2; x4*x };
            
            const MATERIAL_IOR : f32 = 1.5;
            const USE_SPECULAR : bool = true; // enables both specularity AND internal scattering
            
            // if both these are false, a uniform hemisphere is used
            const USE_LAMBERT_COSINE_LAW : bool = true;
            const USE_SUPER_COSINE : bool = false;
            
            const USE_INTERNAL_FOG : bool = false; // use internal "fog" occlusion and rescattering
            
            // the following two configs pass, or are close to passing, the "flat grey ball" test
            // (white furnace without a white background)
            // - cosine law, no internal fog, yes specular
            // - cosine law, no internal fog, no specular
            // - uniform hemisphere, yes internal fog, yes specular
            
            let mut light_t = lightdir.cross(vec3(1.0, 2.22, 3.3331515313)).normalize();
            let light_b = lightdir.cross(light_t).normalize();
            light_t = lightdir.cross(light_b);
            let light_mat = mat3(lightdir, light_t, light_b);
            
            for _i in 0..(count as usize)
            {
                let _a = rng_f32(&mut rngstate) * PI * 2.0;
                let _braw = rng_f32(&mut rngstate);
                
                let _b = _braw.sqrt();   // energyweight: 1.0
                //let _b = _braw;          // energyweight: _braw * 2.0
                //let _b = (1.0 - (_braw - 1.0).powi(2)).sqrt(); // energyweight: 2.0 * (1.0 - _braw)
                //let mut samp_energyweight = 2.0 * (1.0 - _braw);
                let mut samp_energyweight = 1.0;
                let mut z = _a.sin() * _b;
                let mut y = _a.cos() * _b;
                
                let mut x = -(1.0 - y*y - z*z).sqrt();
                let orig_x = x;
                let _tmp = light_mat * vec3(x, y, z);
                x = _tmp.x;
                y = _tmp.y;
                z = _tmp.z;
                
                if x != x { continue; }
                
                let normal = vec3(x, y, z).normalize();
                
                // Light ray is extremely pointed away from the camera and unlikely
                //  to come in through the aperture it even after DOF
                if normal.z < -0.5 { continue; }
                
                let mut t = normal.cross(Vec3::X - Vec3::Z).normalize();
                let b = normal.cross(t).normalize();
                t = normal.cross(b);
                let tbn = mat3(t, b, normal);
                
                let f90 = 1.0;
                let f0_s = (MATERIAL_IOR-1.0)/(MATERIAL_IOR+1.0);
                let f0 = f0_s*f0_s;
                let fschlick = schlick_approx(-lightdir.dot(normal)) * (f90-f0) + f0;
                
                #[inline(always)]
                fn randsph_from_z(tbn : &Mat3, rngstate : &mut u64, z : f32) -> Vec3
                {
                    let r1 = z;
                    let r2 = rng_f32(rngstate);
                    let rz = r1;
                    //let theta = rz.acos();
                    let thetasin = (1.0 - rz*rz).sqrt();
                    let phi = r2 * (2.0 * PI);
                    let rx = thetasin * phi.cos();
                    let ry = thetasin * phi.sin();
                    
                    let rvec_ws = tbn * vec3(rx, ry, rz);
                    rvec_ws
                }
                
                #[inline(always)]
                fn randsph(tbn : &Mat3, rngstate : &mut u64) -> Vec3
                {
                    let r1 =
                    if USE_LAMBERT_COSINE_LAW
                    {
                        rng_f32(rngstate).sqrt() // cosine law
                    }
                    else if USE_SUPER_COSINE
                    {
                        rng_f32(rngstate).sqrt().sqrt().sqrt()
                    }
                    else
                    {
                        rng_f32(rngstate) // uniform hemispherical
                    };
                    randsph_from_z(tbn, rngstate, r1)
                }
                
                let mut scatter_dir = vec3(0.0, 0.0, 0.0);
                
                let do_specular = USE_SPECULAR && rng_f32(&mut rngstate).fract() < fschlick;
                if do_specular
                {
                    scatter_dir = lightdir.reflect(normal);
                    scatter_dir = scatter_dir.normalize();
                }
                else
                {
                    if !USE_SPECULAR
                    {
                        scatter_dir = randsph(&tbn, &mut rngstate);
                    }
                    else
                    {
                        let mut scatter_z = rng_f32(&mut rngstate); // uniform hemispherical
                        
                        let mut incount = 128;
                        while incount > 0
                        {
                            if USE_LAMBERT_COSINE_LAW
                            {
                                scatter_z = scatter_z.sqrt(); // cosine law hemispherical
                            }
                            else if USE_SUPER_COSINE
                            {
                                scatter_z = scatter_z.sqrt();
                                scatter_z = scatter_z.sqrt();
                                scatter_z = scatter_z.sqrt();
                            }
                            
                            let ndot = scatter_z; // 1.0 when straight out
                            
                            // 0.69 is our material's typical opacity when going "straight out"
                            //   from the typical particle scattering location. 0.69 seems plausible: 50% chance of reabsorption
                            //   when trying to go straight out (e^-0.69 ~= 0.5) from the typical emission point.
                            //let npath = (scatter_dir / ndot).length();
                            //let npath = 1.0 / ndot; // accurate
                            //let occlusion = 1.0 - (-0.69 * npath).exp();
                            let occlusion = 1.0_f32.min(1.05-ndot*0.55); // fast approximation
                            //let occlusion = 1.0 - (-0.69 * npath).exp();
                            if USE_INTERNAL_FOG && rng_f32(&mut rngstate) < occlusion
                                { scatter_z = rng_f32(&mut rngstate); incount -= 1; continue; }
                            
                            //float out_angle = asin(sin(acos(ndot))*1.5);
                            //float out_angle = asin(sqrt(1.0-ndot*ndot)*1.5);
                            // 1.5 is our material's IOR. 1.5 is typical.
                            //if ndot < 0.745355992
                            //    { scatter_dir = randsph(&tbn, &mut rngstate); incount -= 1; continue; }
                            let out_sin = (1.0f32-ndot*ndot).sqrt()*MATERIAL_IOR; // 0.0 when straight out
                            //if out_sin >= 1.0
                            //    { scatter_dir = randsph(&tbn, &mut rngstate); incount -= 1; continue; }
                            let out_cos = (1.0f32 - out_sin*out_sin).sqrt(); // 1.0 when straight out
                            let fresnel = schlick_approx(out_cos); // 0.04 when straight out
                            if USE_SPECULAR && out_cos != out_cos || rng_f32(&mut rngstate) < fresnel
                                { scatter_z = rng_f32(&mut rngstate); incount -= 1; continue; }
                            
                            scatter_dir = randsph_from_z(&tbn, &mut rngstate, scatter_z);
                            let flatty = scatter_dir.reject_from(normal);
                            scatter_dir = flatty.normalize() * out_sin + normal * out_cos;
                            
                            break;
                        }
                        if incount == 0 { continue; }
                    }
                }
                
                let energy = 1.0;
                // focus debug grid
                //let energy = ((z*5.0).round() + (y*5.0).round() + 0.5 + 20.0) % (2.0) * 0.6 + 0.4;
                
                // note that the specular lobe will always focus at a different focal length than the ball
                //    when using a large aperture, and this is physically correct
                
                if scatter_dir.z > 0.0
                {
                    let mut scatter_dir = scatter_dir.as_dvec3().normalize();
                    let pos = normal.as_dvec3().normalize();
                    //let lens_z = 30.0 + 0.25;
                    //let lens_z = 100.0 + 0.5;
                    let lens_z = 300.0 + 0.7;
                    //let lens_z = 600.0 + 0.5;
                    //let lens_z = 3000.0 + 0.5;
                    let tz = lens_z - pos.z;
                    let on_lens = pos + scatter_dir / scatter_dir.z * tz;
                    
                    ////////////
                    // various lense configurations that i tested before settling on one
                    ////////////
                    
                    // aperture
                    //if on_lens.x*on_lens.x + on_lens.y*on_lens.y > 2.0 { continue; }
                    //if on_lens.x*on_lens.x + on_lens.y*on_lens.y > 20.0 { continue; }
                    //if on_lens.x*on_lens.x + on_lens.y*on_lens.y > 200.0 { continue; }
                    //if on_lens.x*on_lens.x + on_lens.y*on_lens.y > 100.0 { continue; }
                    //samp_energyweight *= 2.0;
                    if on_lens.x*on_lens.x + on_lens.y*on_lens.y > 50.0 { continue; }
                    samp_energyweight *= 4.0; // ISO adjustment to make up for small aperture
                    //if on_lens.x*on_lens.x + on_lens.y*on_lens.y > 400.0 { continue; }
                    //samp_energyweight *= 2.0;
                    //if on_lens.x*on_lens.x + on_lens.y*on_lens.y > 20000.0 { continue; }
                    
                    // camera lens ior, not material, do not change
                    let ior = 1.5;
                    
                    //let sphericalness = 0.0665;
                    //let sensor_dist_from_lens = lens_z + 0.25;
                    //let imgscale = 1.0;
                    //
                    //let sphericalness = 0.05;
                    //let sensor_dist_from_lens = lens_z * 2.0 + 1.5;
                    //let imgscale = 0.5;
                    //
                    //let sphericalness = 0.0355;
                    //let sensor_dist_from_lens = lens_z * 20.0;
                    //let imgscale = 0.05;
                    //
                    //let sphericalness = 0.03405;
                    //let sensor_dist_from_lens = lens_z * 200.0;
                    //let imgscale = 0.005;
                    
                    //let sphericalness = 0.01003;
                    //let sensor_dist_from_lens = lens_z * 200.0;
                    //let imgscale = 0.005;
                    
                    let sphericalness = 0.0033455;
                    let sensor_dist_from_lens = lens_z * 200.0;
                    let imgscale = 0.005;
                    
                    //let sphericalness = 0.001673;
                    //let sensor_dist_from_lens = lens_z * 200.0;
                    //let imgscale = 0.005;
                    
                    //let sphericalness = 0.00033495;
                    //let sensor_dist_from_lens = lens_z * 200.0;
                    //let imgscale = 0.005;
                    
                    ////////////
                    // end lense configurations
                    ////////////
                    
                    // flat lens entry surface
                    let mut normal = dvec3(-on_lens.x * sphericalness, -on_lens.y * sphericalness, 0.0);
                    normal.z = (1.0 - normal.dot(normal)).sqrt();
                    if normal.z != normal.z { continue; }
                    
                    let ndot = scatter_dir.dot(normal); // 1.0 if no shift
                    let out_sin = (1.0f64-ndot*ndot).sqrt()/ior; // 0.0 if no shift
                    let out_cos = (1.0f64 - out_sin*out_sin).sqrt(); // 1.0 if no shift
                    let flatty = scatter_dir.reject_from(normal);
                    scatter_dir = flatty.normalize() * out_sin + normal * out_cos;
                    
                    normal.x = -normal.x;
                    normal.y = -normal.y;
                    
                    // flat lens exit surface
                    let ndot = scatter_dir.dot(normal);
                    let out_sin = (1.0f64-ndot*ndot).sqrt()*ior;
                    if out_sin >= 1.0 { continue; }
                    let out_cos = (1.0f64 - out_sin*out_sin).sqrt();
                    let flatty = scatter_dir.reject_from(normal);
                    scatter_dir = flatty.normalize() * out_sin + normal * out_cos;
                    
                    let mut on_sensor = on_lens + scatter_dir / scatter_dir.z * sensor_dist_from_lens;
                    on_sensor *= 0.8 * imgscale;
                    if on_sensor.x != on_sensor.x { continue; }
                    if on_sensor.y != on_sensor.y { continue; }
                    
                    // sensor
                    let xc = ((on_sensor.x * 0.5 + 0.5) * WIDTH as f64) as isize;
                    let yc = ((on_sensor.y * 0.5 + 0.5) * WIDTH as f64) as isize;
                    if xc >= WIDTH as isize || yc >= WIDTH as isize || xc < 0 || yc < 0 { continue; }
                    let xc = xc as usize;
                    let yc = yc as usize;
                    
                    let out = &mut data[yc * WIDTH as usize + xc];
                    //*out += 30.0 * energy / (-x + 0.003); 
                    //*out += 30.0 * energy / (-orig_x + 0.0001) * samp_energyweight;
                    *out += 30.0 * energy * samp_energyweight;
                }
            }
        });
        for data in mv
        {
            for i in 0..256*256
            {
                self.field[i] += data[i];
            }
        }
        rng_f32(&mut rngstate);
        self.rngstate = rngstate;
        self.ncount += count * chunkn as u64;
    }
}

fn main() -> Result<(), Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64 * 3.0, WIDTH as f64 * 3.0);
        #[allow(deprecated)]
        Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Tracer Marill")
                        .with_inner_size(size)
                )
                .unwrap(),
        )
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH, WIDTH, surface_texture)?
    };
    let mut world = World::new();

    #[allow(deprecated)]
    let res = event_loop.run(|event, elwt| {
        match event {
            Event::Resumed => {}
            Event::NewEvents(_) => input.step(),
            Event::AboutToWait => input.end_step(),
            Event::DeviceEvent { event, .. } => input.process_device_event(&event),
            Event::WindowEvent { event, .. } =>
            {
                if input.process_window_event(&event)
                {
                    if input.key_pressed(KeyCode::Escape) || input.close_requested()
                    {
                        elwt.exit();
                        return;
                    }

                    if let Some(size) = input.window_resized()
                        && size.width > 0
                        && size.height > 0
                        && let Err(_) = pixels.resize_surface(size.width, size.height)
                    {
                        elwt.exit();
                        return;
                    }
                }

                fn to_srgb(mut c : f32) -> f32
                {
                    c = c.max(0.0);
                    if c < 0.0031308
                    {
                        return c * 12.92;
                    }
                    //vec3 higher = vec3(1.055) * pow(c, vec3(1.0/2.4)) - vec3(0.055);
                    1.055 * c.powf(1.0/2.4) - 0.055
                }

                if event == WindowEvent::RedrawRequested
                {
                    world.continue_trace();

                    for (i, pixel) in pixels.frame_mut().chunks_exact_mut(4).enumerate() {
                        let mut x = (i % WIDTH as usize) as usize;
                        x = WIDTH as usize - 1 - x;
                        let mut y = (i / WIDTH as usize) as usize;
                        y = WIDTH as usize - 1 - y;
                        
                        let norm = (WIDTH as f32*WIDTH as f32) / world.ncount as f32;

                        let val = world.field[y * WIDTH as usize + x] * norm;//.powf(1.0/2.2);
                        let val = to_srgb(val);
                        let val = (val * 255.0 + 0.5) as u8;
                        let rgba = [val, val, val, 0xff];

                        pixel.copy_from_slice(&rgba);
                    }
                
                    if let Err(_) = pixels.render()
                    {
                        elwt.exit();
                        return;
                    }
                    
                    let next = std::time::Instant::now();
                    let dt = world.t_prev.elapsed().as_secs_f64();
                    world.t_prev = next;
                    //world.dt_mem = world.dt_mem.lerp(dt, 0.05);
                    world.dt_mem = world.dt_mem.lerp(dt, 1.0 - (-1.5 * dt).exp());

                    window.set_title(&format!("Tracer Marill {:.02}fps / {}mil rays", 1.0/world.dt_mem, world.ncount/1000000));
                    window.request_redraw();
                }
            }
            _ => {}
        }
    });
    res.map_err(|e| Error::UserDefined(Box::new(e)))
}
