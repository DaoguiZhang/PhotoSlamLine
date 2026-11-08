# TUM fr3_long_office_household crash — root cause (KF 2, free(): invalid pointer)

## Symptom
`bin/tum_rgbd` aborts (134) right after
`LocalBundleAdjustmentWithLine_Optimization_Plucker_Reg` for KF 2, before any
loop/GBA (0 `Starting Global Bundle Adjustment`, 0 `SyncEvent`).

## GDB evidence (`reports/gdb_crash.log`)
Crashing thread (Thread 3, LocalMapping `RunWithLine`):

```
#6  malloc_printerr ("free(): invalid pointer")
#8  __GI___libc_free
#9  ORB_SLAM3::MappingOperation::~MappingOperation()
      from /workspace/code/PSL-perf/lib/libgaussian_mapper.so
#10 ORB_SLAM3::LocalMapping::RunWithLine()
      from /workspace/code/PSL-perf/ORB-SLAM3/lib/libORB_SLAM3.so
```

## Cause: ODR / type-layout mismatch across two shared libs
`Atlas.h` (`MappingOperation`) gained `mvGlobalBAPoses`, `mnEventId`,
`mnMapGen` in `b2e997a5d9`. The two DSOs that instantiate the inline
ctor/dtor were built from different header revisions:

| artifact | mtime | header revision |
|---|---|---|
| `ORB-SLAM3/include/Atlas.h` | 2026-10-17 09:18:49 | new |
| `ORB-SLAM3/build/.../LocalMapping.cc.o` | 2026-10-16 13:38:23 | **old** |
| `ORB-SLAM3/build/.../Optimizer.cc.o` | 2026-10-16 13:40:33 | **old** |
| `ORB-SLAM3/lib/libORB_SLAM3.so` | 2026-10-16 13:40:33 | **old** |
| `lib/libgaussian_mapper.so` | 2026-10-17 09:19:08 | new |

Inline symbols confirm different layouts:

```
libgaussian_mapper.so      W MappingOperation::~MappingOperation()  0x24d bytes  (new)
ORB-SLAM3/lib/libORB_SLAM3.so W MappingOperation::~MappingOperation() 0x22d bytes (old)
ORB-SLAM3/lib/libORB_SLAM3.so W MappingOperation::MappingOperation(MappingOperation const&) 0xec5
```

`LocalMapping.cc` (old layout) allocates/constructs the object; the weak
inline destructor binds to the **first loaded** definition, i.e. the new one
in `libgaussian_mapper.so` (loaded before `libORB_SLAM3.so`). The new
destructor reads `mvGlobalBAPoses` beyond the end of the old-layout object
and calls `free()` on garbage pointers.

The `GlobalBA` consumer branch never ran; the failure is unconditional ODR
breakage caused by the stale ORB-SLAM3 build, not the event logic.

## Fix
Rebuild `ORB-SLAM3/build` with the current header (same cached Release
configuration), then relink the top-level tree:

```
cmake --build ORB-SLAM3/build -j8
cmake --build build -j8
```
