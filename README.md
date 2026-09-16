# Five-Axis FFF Benchmark Code

This repository contains the Python code used to generate, solve, visualize and convert three five-axis FFF benchmark toolpaths:

1. **Bent pipe benchmark**
2. **Femoral joint benchmark**
3. **Skull flap implant benchmark**

The three benchmarks use the same general workflow:

```text
input geometry / sliced path
        |
        v
S01 - generate CL data
        |
        v
S02 - solve inverse kinematics
        |
        +----> S03 - visualize solved motion (optional)
        |
        v
S04 - generate printer G-code
```

Each benchmark is self-contained in its own folder. Shared settings for that benchmark are stored in its `S00_..._parameters.py` file.

---

## Repository structure

```text
FiveAxis project code/
|
|-- README.md
|-- environment.yml
|
|-- Bent pipe benchmark/
|   |-- S00_bent_pipe_parameters.py
|   |-- S01_bent_pipe_cldata_generator.py
|   |-- S02a_bent_pipe_ik_solver_numerical.py
|   |-- S02b_bent_pipe_ik_solver_hybrid.py
|   |-- S03_bent_pipe_visualizer.py
|   `-- S04_bent_pipe_gcode_generator.py
|
|-- Femoral joint benchmark/
|   |-- S00_femoral_joint_parameters.py
|   |-- S01_femoral_joint_cldata_generator.py
|   |-- S02a_femoral_joint_ik_solver_numerical.py
|   |-- S02b_femoral_joint_ik_solver_analytical.py
|   |-- S03_femoral_joint_visualizer.py
|   |-- S04_femoral_joint_gcode_generator.py
|   `-- femoral_joint_mandrel_cura.gcode
|
`-- Skull flap implant benchmark/
    |-- S00_skull_implant_parameters.py
    |-- S01_skull_implant_cldata_generator.py
    |-- S02a_skull_implant_ik_solver_numerical.py
    |-- S02b_skull_implant_ik_solver_analytical.py
    |-- S03_skull_implant_visualizer.py
    |-- S04_skull_implant_gcode_generator.py
    `-- skull_implant_atomslicer.gcode
```

Generated `.txt`, `.mp4` and `.gcode` files are written into the corresponding benchmark folder.

---

## Python environment

The required Conda environment is defined in `environment.yml`.

Create it with:

```bash
conda env create -f environment.yml
```

Then activate it:

```bash
conda activate printer5axis
```

The project was developed with **Python 3.10**. The environment contains the numerical, plotting, robotics and 3D-visualization packages required by the scripts.

The scripts can be run from a terminal or from Spyder. When using Spyder, it is easiest to use the corresponding benchmark folder as the working directory.

---

# General workflow

The numbered scripts indicate their place in the pipeline.

## S00 - Parameters

`S00_..._parameters.py` contains the benchmark-specific settings and file paths.

This includes, where applicable:

- machine geometry;
- coordinate conventions;
- toolhead geometry;
- CL-data generation settings;
- IK tolerances and branch settings;
- printer coordinate mapping;
- machine limits;
- temperatures, speeds and extrusion settings.

The file paths are based on the location of the parameter file itself. No absolute project path needs to be configured when the repository is moved to another computer.

Most benchmark settings should be changed in `S00` rather than inside the other pipeline scripts. Display-only visualizer settings remain in `S03` where applicable.

## S01 - CL-data generation

`S01` generates the toolpath used by the IK solver.

CL-data rows have the form:

```text
x  y  z  i  j  k
```

where `x y z` describe the required tool-tip position and `i j k` describe the associated orientation vector.

The exact source of the CL data differs between the three benchmarks and is described below.

## S02 - Inverse kinematics

The IK stage converts each CL-data row into printer machine coordinates:

```text
x  y  z_table  B  C
```

Two IK implementations are included:

- `S02a` - numerical solver;
- `S02b` - analytical or hybrid solver.

They are **alternative implementations of the same pipeline stage**. It is not necessary to run both.

Both write to the same `02_..._solved_ik.txt` file, so running the second solver will replace the result of the first. If the two methods are being compared, save or rename the first result before running the other solver.

For normal use, the `S02b` implementation is the intended main solver; `S02a` is retained as a numerical comparison/reference implementation.

## S03 - Visualization

`S03` reconstructs the printer geometry in PyVista and animates the solved `x y z_table B C` path.

This step is optional and is intended to check:

- machine motion;
- B/C orientation;
- nozzle-tip trajectory;
- general path placement.

The visualizer does not modify the solved IK data or the final G-code.

Depending on the local visualizer settings, it can run interactively or save an MP4 animation.

## S04 - G-code generation

`S04` converts the solved IK path into Klipper-compatible five-axis G-code.

This stage applies the benchmark-specific printer coordinate mapping, limits, speeds, temperatures and extrusion handling.

Generated print files use numbered filenames such as:

```text
04_bent_pipe_5axis_print_1.gcode
04_bent_pipe_5axis_print_2.gcode
```

This prevents an earlier generated G-code file from being overwritten.

---

# 1. Bent pipe benchmark

The bent-pipe benchmark is generated completely in Python and does not require an external sliced input file.

## Pipeline

```text
S00_bent_pipe_parameters.py
        |
        v
S01_bent_pipe_cldata_generator.py
        |
        v
01_bent_pipe_cldata.txt
        |
        v
S02a numerical  OR  S02b hybrid IK
        |
        v
02_bent_pipe_solved_ik.txt
        |
        +----> S03_bent_pipe_visualizer.py
        |
        v
S04_bent_pipe_gcode_generator.py
        |
        v
04_bent_pipe_5axis_print_N.gcode
```

### S01

The CL-data generator constructs a bent centerline consisting of straight and curved sections and builds a continuous helical toolpath around it.

The generated path includes both tool-tip position and orientation information.

Output:

```text
01_bent_pipe_cldata.txt
```

### S02

- `S02a_bent_pipe_ik_solver_numerical.py` solves B/C numerically.
- `S02b_bent_pipe_ik_solver_hybrid.py` uses analytical B/C solving where possible and retains a numerical fallback.

Output:

```text
02_bent_pipe_solved_ik.txt
```

### S03

Visualizes the solved machine motion and nozzle-tip trace.

Optional animation output:

```text
03_bent_pipe_visualization.mp4
```

### S04

Calculates extrusion from CL-path segment length and generates the five-axis printer G-code.

Output:

```text
04_bent_pipe_5axis_print_N.gcode
```

---

# 2. Femoral joint benchmark

The femoral-joint benchmark combines a conventionally sliced cylindrical mandrel with non-planar dome layers.

Required input:

```text
femoral_joint_mandrel_cura.gcode
```

This is the sliced gcode of a 3D model of a femoral joint analogue. The model was sliced using Ultimaker's Cura slicer.

## Pipeline

```text
femoral_joint_mandrel_cura.gcode
        |
        v
S01_femoral_joint_cldata_generator.py
        |
        |-- 01a_femoral_joint_mandrel_cldata.txt
        |-- 01b_femoral_joint_dome_cldata.txt
        |-- dome travel indices
        `-- 01c_femoral_joint_combined_cldata.txt
                    |
                    v
       S02a numerical OR S02b analytical IK
                    |
                    v
       02_femoral_joint_solved_ik.txt
                    |
                    +----> S03_femoral_joint_visualizer.py
                    |
                    v
       S04_femoral_joint_gcode_generator.py
                    |
                    v
       04_femoral_joint_5axis_print_N.gcode
```

### S01

This script performs two tasks:

1. converts the Cura mandrel G-code to conventional vertical-tool CL data;
2. generates the non-planar hemisphere/dome spiral directly in Python.

The two paths are combined into one CL-data file.

Separate no-extrusion index files identify safe repositioning moves between dome layers. These indices are used by `S04` so that repositioning moves are written as travel moves rather than printing moves.

Main generated files:

```text
01a_femoral_joint_mandrel_cldata.txt
01b_femoral_joint_dome_cldata.txt
01b_femoral_joint_dome_no_extrude_indices.txt
01c_femoral_joint_combined_cldata.txt
01c_femoral_joint_combined_no_extrude_indices.txt
```

### S02

- `S02a_femoral_joint_ik_solver_numerical.py` provides the numerical IK implementation.
- `S02b_femoral_joint_ik_solver_analytical.py` provides the analytical implementation with numerical fallback where required.

Output:

```text
02_femoral_joint_solved_ik.txt
```

### S03

Visualizes the mandrel and dome sections of the solved five-axis path.

Optional animation output:

```text
03_femoral_joint_visualization.mp4
```

Display-only geometry corrections in this file do not change the IK result or generated G-code.

### S04

Generates the final G-code for both the mandrel and dome sections.

The generator:

- uses separate mandrel and dome print settings;
- uses the no-extrusion indices for inter-layer travel;
- can insert the configured filament change at the mandrel-to-dome transition.

Output:

```text
04_femoral_joint_5axis_print_N.gcode
```

---

# 3. Skull flap implant benchmark

The skull-flap benchmark starts from an .stl 3D model of a skull-flap implant. This 3D model was then sliced for multi-axis printing using AtomSlicer. This results in G-code containing both the 3D path and surface-normal information.

Required input:

```text
skull_implant_atomslicer.gcode
```

The source G-code must contain normal-vector comments associated with the path, for example:

```text
G1 X... Y... Z... E... ; (i, j, k)
```

## Pipeline

```text
skull_implant_atomslicer.gcode
        |
        v
S01_skull_implant_cldata_generator.py
        |
        v
01_skull_implant_cldata.txt
        |
        v
S02a numerical OR S02b analytical IK
        |
        v
02_skull_implant_solved_ik.txt
        |
        +----> S03_skull_implant_visualizer.py
        |
        v
S04_skull_implant_gcode_generator.py
        |
        +---- reads original AtomSlicer extrusion values
        |
        v
04_skull_implant_5axis_print_N.gcode
```

### S01

The AtomSlicer parser:

- reads XYZ motion;
- reads the `(i, j, k)` normal-vector comment;
- normalizes the orientation vector;
- optionally removes duplicate or non-extrusion rows;
- applies the configured centering/offset settings;
- writes the resulting CL data.

Output:

```text
01_skull_implant_cldata.txt
```

### S02

- `S02a_skull_implant_ik_solver_numerical.py` is the numerical comparison solver.
- `S02b_skull_implant_ik_solver_analytical.py` is the preferred solver for the skull implant path.

Output:

```text
02_skull_implant_solved_ik.txt
```

### S03

Visualizes the solved machine path and nozzle-tip trajectory.

Optional animation output:

```text
03_skull_implant_visualization.mp4
```

The `VIZ_PATH_OFFSET_X_MM`, `VIZ_PATH_OFFSET_Y_MM` and `VIZ_PATH_OFFSET_Z_MM` settings are **visualization-only offsets**. They change where the path is drawn in the visualizer and do not change the solved IK or final printer G-code.

### S04

The skull G-code generator differs from the other benchmarks because extrusion is not recalculated from path length.

Instead, it rereads:

```text
skull_implant_atomslicer.gcode
```

and maps the original AtomSlicer extrusion values to the corresponding CL/IK rows.

The solved XYZ path is then placed within the configured machine build area. B and C are not changed by this placement.

Output:

```text
04_skull_implant_5axis_print_N.gcode
```

---


# Generated files

The intermediate `.txt` files are deterministic pipeline outputs and can be regenerated by rerunning the corresponding stage.

In general:

```text
01...   CL-data generation
02...   solved inverse kinematics
03...   visualization / animation
04...   final printer G-code
```

Stages `S01` and `S02` use fixed output filenames and will replace their previous generated output.

Stage `S04` creates a new numbered G-code file rather than overwriting an earlier print file.

---

# Notes

- Run scripts from their own benchmark folder so the local `S00_..._parameters.py` module is available.
- Only one `S02` solver (a or b) needs to be run for a normal pipeline execution.
- The numerical and analytical/hybrid solvers intentionally write to the same solved-IK filename.
- The visualizers are verification tools and do not alter the pipeline data.
- Machine-specific limits and print settings should be checked in the relevant `S00` parameter file before generating (and using) printer G-code.
