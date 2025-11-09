# Dialog Generation Pipeline - Extensions

## Overview

This directory contains the dialog generation pipeline for ProAssist, with several key extensions:

1. **OpenRouter API Integration**: Replace local vLLM inference with cloud-based LLM APIs
2. **EgoExo4D Dataset Support**: Dialog generation for EgoExo4D cooking videos
3. **EPFL Dataset Support**: Dialog generation for EPFL egocentric cooking videos
4. **Keystep-Aware Progress Summaries**: Enhanced assistant summaries using keystep annotations

## 1. OpenRouter API Integration

### Overview

Added support for cloud-based LLM inference via OpenRouter API, replacing the need for local vLLM deployment. This enables dialog generation without GPU resources.

### Key Features

- **API-based inference**: Use OpenRouter to access models like Gemini, Claude, GPT, etc.
- **Backward compatible**: Drop-in replacement for existing `LLMGenerator` interface
- **Multimodal support**: Handles both text-only and vision-language inputs
- **Rate limiting**: Automatic retry logic for API rate limits
- **Batch processing**: Sequential processing with configurable delays

### Components

**`openrouter_utils.py`**:

- **`OpenRouterConfig`**: Configuration dataclass for API settings
  - API key, model name, base URL, sampling parameters
  
- **`OpenRouterGenerator`**: Core API client
  - `build()`: Initialize from model ID and API key
  - `generate()`: Single conversation generation
  - `batch_generate()`: Multiple conversations with rate limit handling
  - Supports multimodal content (text + base64-encoded images)
  
- **`LLMGenerator`**: Compatibility wrapper
  - Maintains existing interface for seamless integration
  - Filters vLLM-specific parameters
  - Default sampling args: `temperature=0.5`, `top_p=0.95`, `max_tokens=4096`

### Usage

```bash
# Set API key
export OPENROUTER_API_KEY="your-api-key"

# Use with EgoExo4D generation
python -m mmassist.datasets.generate.generate_egoexo4d \
    --llm google/gemini-2.5-flash \
    --data_dir /path/to/annotations \
    --output_dir /path/to/output

# Use with EPFL generation
python -m mmassist.datasets.generate.generate_epfl \
    --llm anthropic/claude-3.5-sonnet \
    --data_dir /path/to/annotations \
    --output_dir /path/to/output
```

### API Key Setup

```bash
# Required: Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-v1-..."

# Get an API key at: https://openrouter.ai/
```

### Supported Models

Any model available on OpenRouter, including:
- `google/gemini-2.5-flash` (default, fast and cost-effective)
- `google/gemini-2.5-pro` (more capable, for complex tasks)
- `anthropic/claude-3.5-sonnet`
- `openai/gpt-4-turbo`

## 2. EgoExo4D Dataset Support

### Overview

Added complete pipeline for generating dialogs from EgoExo4D cooking video annotations.

### Key Features

- **Atomic action annotations**: Parse timestamped action descriptions
- **Keystep integration**: Load and incorporate keystep annotations
- **Task metadata**: Extract task names from `takes.json`
- **Aria camera frames**: Support for frame-based generation with Aria camera views
- **Clip splitting**: Automatic splitting based on `max_num_lines_per_gen`
- **Frame incorporation**: Three modes for visual information (none/video/descriptions)

### Components

**`generate_egoexo4d.py`**:

- **`load_atomic_descriptions()`**: Load timestamped action annotations
- **`load_keystep_annotations()`**: Load keystep segments
- **`load_take_uid_to_task_name_mapping()`**: Extract task names from takes.json
- **`find_aria_arrow_file()`**: Locate Aria camera frame files
- **`parse_egoexo4d_annotations()`**: Convert to `ParsedVideoAnns` format
- **`load_egoexo4d_dataset()`**: Load all annotations for specified splits
- **`run_local_jobs()`**: Sequential processing with caching
- **`save_local_results()`**: Save with auto-evaluation

### Usage

```bash
# Basic usage
python -m mmassist.datasets.generate.generate_egoexo4d \
    --data_dir /path/to/egoexo4d/annotations \
    --frames_dir /path/to/egoexo4d/frames \
    --splits train,val \
    --output_dir ./output/egoexo4d

# With keystep-aware summaries and frame descriptions
python -m mmassist.datasets.generate.generate_egoexo4d \
    --data_dir /path/to/annotations \
    --frames_dir /path/to/frames \
    --output_dir ./output \
    --use_keysteps True \
    --use_frames descriptions \
    --frame_desc_llm google/gemini-2.5-flash
```

### Command-Line Arguments

- `--data_dir`: EgoExo4D annotations directory
- `--frames_dir`: Directory with frame Arrow files
- `--splits`: Comma-separated splits (e.g., `train,val`)
- `--llm`: LLM model for dialog generation
- `--use_keysteps`: Enable keystep-aware progress summaries
- `--use_frames`: Frame mode (`none`, `video`, `descriptions`)
- `--max_num_lines_per_gen`: Max atomic actions per clip (default: 50)

### Data Format

**Input**:
- `atomic_descriptions_{split}_filtered_longest_keysteps.json`: Timestamped actions
- `keystep_{split}.json`: Keystep segments with timing
- `takes.json`: Task metadata (task names, take names)
- Frame Arrow files: `{take_name}_downscaled_*aria*.arrow`

**Output**:
- Per-video JSON files: `{split}/{take_uid}.json`
- Split-level aggregated files: `{split}.json`

## 3. EPFL Dataset Support

### Overview

Added complete pipeline for generating dialogs from EPFL egocentric cooking video annotations.

### Key Features

- **Hierarchical annotations**: Combines coarse activities and fine-grained actions
- **Confusion filtering**: Automatically filters out confused annotations
- **Verb-noun actions**: Constructs action descriptions from verb and noun pairs
- **Multi-camera support**: Uses HoloLens camera frames
- **Frame descriptions**: Optional LLM-based frame descriptions for each action

### Components

**`generate_epfl.py`**:

- **`load_epfl_annotations()`**: Load coarse and fine-grained annotations
  - Reads `activity_annotations.json` (coarse activities)
  - Reads `actions_annotations.xlsx` (fine-grained actions)
  - Filters out confused annotations (`Confusion != 1`)
  
- **`combine_annotations()`**: Group fine-grained actions under coarse activities
  - Hierarchical structure: activities contain actions
  - Temporal overlap matching
  
- **`parse_epfl_annotations()`**: Convert to `ParsedVideoAnns` format
  - Creates timestamped descriptions
  - Optional frame description generation
  - Splits into clips based on line count
  
- **`load_epfl_dataset()`**: Load all annotations for specified splits
- **`run_local_jobs()`**: Sequential processing with caching
- **`save_local_results()`**: Save with auto-evaluation

### Usage

```bash
# Basic usage
python -m mmassist.datasets.generate.generate_epfl \
    --data_dir /path/to/epfl/annotations \
    --frames_dir /path/to/epfl/frames \
    --splits train,test \
    --output_dir ./output/epfl

# With frame descriptions
python -m mmassist.datasets.generate.generate_epfl \
    --data_dir /path/to/annotations \
    --frames_dir /path/to/frames \
    --output_dir ./output \
    --use_frames descriptions \
    --frame_desc_llm google/gemini-2.5-flash
```

### Command-Line Arguments

- `--data_dir`: EPFL annotations directory
- `--frames_dir`: Directory with frame Arrow files
- `--splits`: Comma-separated splits (e.g., `train,test`)
- `--llm`: LLM model for dialog generation
- `--use_frames`: Frame mode (`none`, `video`, `descriptions`)
- `--max_num_lines_per_gen`: Max lines per clip (default: 20)

### Data Format

**Input** (per session):
- `annotations/activity_annotations.json`: Coarse activity segments
- `annotations/actions_annotations.xlsx`: Fine-grained action annotations
- Frame Arrow files: `{split}_{participant}_{session}_hololens_compressed.arrow`

**Output**:
- Per-video JSON files: `{split}/{video_uid}.json`
- Split-level aggregated files: `{split}.json`

### EPFL Annotation Structure

```
Coarse activity: [30.5s-120.3s] Preparing vegetables
 - [35.2s] cut carrot (image shows: hands chopping orange carrot on board)
 - [48.7s] peel potato (image shows: peeler removing potato skin)
 - [89.1s] dice onion (image shows: diced onion pieces on cutting board)
```

## 4. Keystep-Aware Progress Summaries

### What's New

Enhanced assistant progress summaries that incorporate keystep annotations to provide more structured and essential information. This addresses key limitations of standard summaries:

- **Future-looking**: Includes next steps, not just past/current events
- **Recipe step awareness**: Shows progress (e.g., "Step 3 of 7: Sautéing onions")
- **Reduced granularity**: High-level progress instead of atomic actions
- **Essential focus**: Only critical details affecting future steps

### Usage

```bash
# Standard progress summaries (default)
python -m mmassist.datasets.generate.generate_egoexo4d \
    --data_dir /path/to/annotations \
    --output_dir /path/to/output

# Keystep-aware progress summaries
python -m mmassist.datasets.generate.generate_egoexo4d \
    --data_dir /path/to/annotations \
    --output_dir /path/to/output \
    --use_keysteps True
```

### Summary Structure

Enhanced summaries follow this format:

```
The time elapsed since the start of the task is 45.3 seconds.

DISH: [dish name]
PROGRESS: Step X/Y - [current stage]
CURRENT STATE: [what's happening now]
COMPLETED: [high-level past steps]
NEXT: [what comes next]
TIMERS: [active timers or "None active"]
NOTES: [techniques, deviations, user preferences]
VISUAL: [observable states from recent frames]
```

### Summary Content

Keystep-aware summaries include:

- Current recipe step number and total steps
- Completed steps affecting future actions
- State of ingredients (e.g., "onions translucent")
- Equipment status (what's in pans/bowls/oven)
- Active timers and time-sensitive actions
- Techniques used or deviations from recipe
- User context (skill level, preferences, substitutions)
- Visual verification from recent frames

## Implementation Details

### Key Components

1. **`PROGRESS_SUMMARY_WITH_KEYSTEPS_PROMPT_TEMPLATE`** (`dialog_simulation.py`)
   - Enhanced prompt with task knowledge, keysteps context, and structure guidelines

2. **`find_current_and_next_keystep()`** (`dialog_simulation.py`)
   - Finds keystep containing a timestamp or closest previous/next keystep
   - Returns tuple of (current_keystep, next_keystep)

3. **`add_progress_summary_with_keysteps()`** (`dialog_simulation.py`)
   - Generates structured summaries using keystep context
   - Uses only `step_name` field (avoids verbose/inaccurate fields)

4. **`generate_from_annotation(..., use_keysteps=False)`** (`dialog_simulation.py`)
   - Added `use_keysteps` parameter for backward compatibility
   - Extracts keystep segments from annotation data
   - Falls back to standard summaries if keysteps unavailable

5. **`--use_keysteps`** flag (`generate_egoexo4d.py`)
   - Command-line argument to enable keystep-aware summaries
   - Defaults to `False` for backward compatibility

### Modified Files

- `mmassist/datasets/generate/dialog_simulation.py`: Core prompt and summary functions
- `mmassist/datasets/generate/generate_egoexo4d.py`: EgoExo4D-specific integration

## Backward Compatibility

The original `add_progress_summary()` function is preserved unchanged. The system automatically falls back to standard summaries when:

- `use_keysteps=False` (default)
- Keystep annotations are unavailable
- Keystep segments are missing from annotation

## Design Decisions

- **Simplified keystep usage**: Only uses `step_name` to avoid verbosity and inaccuracy
- **Clean formatting**: `1. Unbox package (0.9s - 33.2s)`
- **Simple references**: "Step 3/7: Sautéing onions" instead of full descriptions
- **No structural changes**: Existing data structures remain unchanged

## Benefits

1. **Better context**: Recipe step numbers and next steps
2. **Essential information**: Focus on what matters for future guidance
3. **Structured format**: Consistent, easy-to-parse summaries
4. **Temporal awareness**: Clear progress markers via keysteps
5. **User-centric**: Includes preferences, skill level, deviations
6. **Future-looking**: Highlights what needs attention next

## Testing

To verify the implementation:

```bash
# Test with keystep summaries on small dataset
python -m mmassist.datasets.generate.generate_egoexo4d \
    --data_dir /path/to/annotations \
    --output_dir ./test_output \
    --use_keysteps True \
    --max_num_lines_per_gen 10
```

Compare generated summaries with standard mode (`--use_keysteps False`) to validate improvements.

## Requirements

- Keystep annotations available in EgoExo4D dataset
- Keystep segments must include: `start_time`, `end_time`, `step_name`
- Fields `step_description` and `is_essential` are intentionally not used

## Full Example

```bash
# EgoExo4D with all features enabled
python -m mmassist.datasets.generate.generate_egoexo4d \
    --data_dir /projects/beto/proassist_data/datasets/egoexo4d/annotations \
    --frames_dir /projects/beto/proassist_data/processed_data/egoexo4d/frames \
    --splits train,val \
    --output_dir ./output/keystep_summaries \
    --llm google/gemini-2.5-flash \
    --user_types no_talk@2,talk_some@4,talk_more@4 \
    --num_repeats 10 \
    --use_keysteps True \
    --use_frames descriptions \
    --frame_desc_llm google/gemini-2.5-flash \
    --max_num_lines_per_gen 50

# EPFL with frame descriptions
python -m mmassist.datasets.generate.generate_epfl \
    --data_dir /projects/beto/proassist_data/datasets/epfl/annotations \
    --frames_dir /projects/beto/proassist_data/processed_data/epfl/frames \
    --splits train,test \
    --output_dir ./output/epfl \
    --llm google/gemini-2.5-flash \
    --user_types no_talk@2,talk_some@4,talk_more@4 \
    --num_repeats 10 \
    --use_frames descriptions \
    --frame_desc_llm google/gemini-2.5-flash \
    --max_num_lines_per_gen 20
```

## Summary of Files

### New Files

1. **`openrouter_utils.py`**: OpenRouter API integration for cloud-based LLM inference
2. **`generate_egoexo4d.py`**: EgoExo4D dataset dialog generation pipeline
3. **`generate_epfl.py`**: EPFL dataset dialog generation pipeline

### Modified Files

1. **`dialog_simulation.py`**: Added keystep-aware progress summary functions
   - `PROGRESS_SUMMARY_WITH_KEYSTEPS_PROMPT_TEMPLATE`
   - `find_current_and_next_keystep()`
   - `add_progress_summary_with_keysteps()`
   - Updated `generate_from_annotation()` with `use_keysteps` parameter

## Common Patterns

### User Types Configuration

All generation scripts support flexible user type configuration:

```bash
--user_types no_talk@2,talk_some@4,talk_more@4
```

This generates:
- 2 dialogs per video with "no_talk" users (minimal interaction)
- 4 dialogs per video with "talk_some" users (moderate interaction)
- 4 dialogs per video with "talk_more" users (verbose interaction)

### Frame Incorporation Modes

Three modes for incorporating visual information:

1. **`none`** (default): Text-only, no visual information
2. **`video`**: Pass video frames directly to vision-language model
3. **`descriptions`**: Generate text descriptions of frames, incorporate into annotations

```bash
# Option 3: Frame descriptions (recommended)
--use_frames descriptions \
--frame_desc_llm google/gemini-2.5-flash
```

### Output Structure

Both pipelines produce consistent output:

```
output_dir/
├── train/
│   ├── video_001.json
│   ├── video_002.json
│   └── ...
├── val/  (or test/)
│   ├── video_101.json
│   └── ...
├── train.json  (aggregated)
└── val.json    (aggregated)
```

Each video JSON contains:
- Generated dialog turns
- Metadata (gen_args, user types, etc.)
- Auto-evaluation metrics
- Original annotation references

