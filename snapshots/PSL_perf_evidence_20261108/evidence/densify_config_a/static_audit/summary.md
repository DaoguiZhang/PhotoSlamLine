# Static config-key audit

- configs: 61
- keys read by `gaussian_mapper_line.cpp`: 66
- room1 missing keys: 16

## Room1 missing keys

| key | read style |
|---|---|
| `Optimization.absolute_max_ratio` | guarded_explicit_default |
| `Optimization.densify_from_iter_` | strict_canonical_legacy_resolver |
| `Optimization.lambda_anisotropy` | guarded_explicit_default |
| `Optimization.line_sample_step` | guarded_explicit_default |
| `Optimization.line_sigma` | guarded_explicit_default |
| `Optimization.line_top_k` | guarded_explicit_default |
| `Optimization.line_view_weight` | guarded_explicit_default |
| `Optimization.max_anisotropy_threshold` | guarded_explicit_default |
| `Optimization.absolute_max_ratio` | guarded_explicit_default |
| `Optimization.densify_from_iter_` | strict_canonical_legacy_resolver |
| `Optimization.lambda_anisotropy` | guarded_explicit_default |
| `Optimization.line_sample_step` | guarded_explicit_default |
| `Optimization.line_sigma` | guarded_explicit_default |
| `Optimization.line_top_k` | guarded_explicit_default |
| `Optimization.line_view_weight` | guarded_explicit_default |
| `Optimization.max_anisotropy_threshold` | guarded_explicit_default |

## Keys missing from every gaussian YAML

- `Optimization.densify_from_iter_` (strict_canonical_legacy_resolver)

## Direct reads missing in any config

- `Camera.z_far` missing in 1/61
- `Camera.z_near` missing in 1/61
- `GausPyramid.do` missing in 1/61
- `GausPyramid.num_sub_levels` missing in 1/61
- `GausPyramid.sub_level_times_of_use` missing in 1/61
- `GaussianViewer.image_scale` missing in 1/61
- `GaussianViewer.image_scale_main` missing in 1/61
- `Mapper.cull_keyframes` missing in 1/61
- `Mapper.depth_cache` missing in 1/61
- `Mapper.inactive_geo_densify` missing in 1/61
- `Mapper.large_rotation_threshold` missing in 1/61
- `Mapper.large_translation_threshold` missing in 1/61
- `Mapper.local_BA_increased_times_of_use` missing in 1/61
- `Mapper.loop_closure_increased_times_of_use_` missing in 1/61
- `Mapper.min_num_initial_map_kfs` missing in 1/61
- `Mapper.new_keyframe_times_of_use` missing in 1/61
- `Mapper.stable_num_iter_existence` missing in 1/61
- `Model.eval` missing in 1/61
- `Model.resolution` missing in 1/61
- `Model.sh_degree` missing in 1/61
- `Model.white_background` missing in 1/61
- `Monocular.inactive_geo_densify_max_pixel_dist` missing in 1/61
- `Optimization.densification_interval` missing in 1/61
- `Optimization.densify_grad_threshold` missing in 1/61
- `Optimization.densify_min_opacity` missing in 1/61
- `Optimization.densify_until_iter` missing in 1/61
- `Optimization.feature_lr` missing in 1/61
- `Optimization.lambda_dssim` missing in 1/61
- `Optimization.max_num_iterations` missing in 1/61
- `Optimization.opacity_lr` missing in 1/61
- `Optimization.opacity_reset_interval` missing in 1/61
- `Optimization.percent_dense` missing in 1/61
- `Optimization.position_lr_delay_mult` missing in 1/61
- `Optimization.position_lr_final` missing in 1/61
- `Optimization.position_lr_init` missing in 1/61
- `Optimization.position_lr_max_steps` missing in 1/61
- `Optimization.prune_big_point_after_iter` missing in 1/61
- `Optimization.rotation_lr` missing in 1/61
- `Optimization.scaling_lr` missing in 1/61
- `Optimization.weight_line_coherence` missing in 3/61
- `Optimization.weight_line_shape_ecc` missing in 3/61
- `Optimization.weight_line_shape_ori` missing in 3/61
- `Pipeline.compute_cov3D` missing in 1/61
- `Pipeline.convert_SHs` missing in 1/61
- `RGBD.max_depth` missing in 1/61
- `RGBD.min_depth` missing in 1/61
- `Record.all_keyframes_record_interval` missing in 1/61
- `Record.keyframe_record_interval` missing in 1/61
- `Record.record_ground_truth_image` missing in 1/61
- `Record.record_loop_ply` missing in 1/61
- `Record.record_loss_image` missing in 1/61
- `Record.record_rendered_image` missing in 1/61
- `Record.training_report_interval` missing in 1/61
- `Stereo.min_disparity` missing in 1/61
- `Stereo.num_disparity` missing in 1/61
