"""Utilities for incorporating frames into dialog generation."""

import os
import io
import base64
import tempfile
from typing import List, Dict, Tuple, Optional
from PIL import Image
import datasets as hf_datasets

from mmassist.datasets.utils.video_utils import str2img, time_to_frame_index
from mmassist.datasets.generate.openrouter_utils import LLMGenerator


def load_frames_from_arrow(arrow_file: str) -> hf_datasets.Dataset:
    """Load frames from an Arrow file.
    
    Args:
        arrow_file: Path to the Arrow file containing frames
        
    Returns:
        HuggingFace dataset with frames
    """
    if not os.path.exists(arrow_file):
        raise FileNotFoundError(f"Arrow file not found: {arrow_file}")
    
    frames_data = hf_datasets.load_dataset(
        "arrow", data_files=arrow_file, split="train"
    )
    return frames_data


def get_frame_at_timestamp(
    frames_data: hf_datasets.Dataset,
    timestamp: float,
    fps: float = 2.0,
) -> Optional[Image.Image]:
    """Get the frame closest to a given timestamp.
    
    Args:
        frames_data: Dataset containing frames
        timestamp: Time in seconds
        fps: Frames per second of the extracted frames
        
    Returns:
        PIL Image or None if index out of range
    """
    frame_idx = time_to_frame_index(timestamp, fps, rounding="round")
    
    if frame_idx < 0 or frame_idx >= len(frames_data):
        return None
    
    frame_str = frames_data[frame_idx]["frame"]
    return str2img(frame_str)


def get_frames_for_clip(
    frames_data: hf_datasets.Dataset,
    start_time: float,
    end_time: float,
    fps: float = 2.0,
) -> List[Image.Image]:
    """Get all frames within a time range.
    
    Args:
        frames_data: Dataset containing frames
        start_time: Start time in seconds
        end_time: End time in seconds
        fps: Frames per second of the extracted frames
        
    Returns:
        List of PIL Images
    """
    start_idx = time_to_frame_index(start_time, fps, rounding="floor")
    end_idx = time_to_frame_index(end_time, fps, rounding="ceil")
    
    # Clamp to valid range
    start_idx = max(0, start_idx)
    end_idx = min(len(frames_data), end_idx)
    
    frames = []
    for idx in range(start_idx, end_idx + 1):
        if idx < len(frames_data):
            frame_str = frames_data[idx]["frame"]
            frames.append(str2img(frame_str))
    
    return frames


def frames_to_video_bytes(
    frames: List[Image.Image],
    fps: float = 2.0,
) -> bytes:
    """Convert a list of frames to a video file in memory.
    
    Args:
        frames: List of PIL Images
        fps: Target frames per second for the video
        
    Returns:
        Video file as bytes (MP4 format)
    """
    import av
    
    if not frames:
        raise ValueError("No frames provided")
    
    # Create a temporary file for the video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Open video container for writing
        container = av.open(tmp_path, mode='w')
        stream = container.add_stream('h264', rate=fps)
        
        # Set video dimensions from first frame
        width, height = frames[0].size
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        
        # Write frames
        for frame_img in frames:
            # Ensure all frames have the same size
            if frame_img.size != (width, height):
                frame_img = frame_img.resize((width, height))
            
            # Convert PIL Image to VideoFrame
            frame = av.VideoFrame.from_image(frame_img)
            
            # Encode and write
            for packet in stream.encode(frame):
                container.mux(packet)
        
        # Flush remaining packets
        for packet in stream.encode():
            container.mux(packet)
        
        container.close()
        
        # Read the video file as bytes
        with open(tmp_path, 'rb') as f:
            video_bytes = f.read()
        
        return video_bytes
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def image_to_base64_data_url(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 data URL.
    
    Args:
        image: PIL Image
        format: Image format (JPEG or PNG)
        
    Returns:
        Base64-encoded data URL
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{img_str}"


def describe_frame_with_llm(
    frame: Image.Image,
    llm: LLMGenerator,
    prompt: str = "You are viewing an egocentric (first-person) video frame of someone cooking in a kitchen. Describe the most important visual information that would help a kitchen assistant provide better guidance. In 2-3 concise sentences (30-50 words), focus on: (1) Key ingredients, tools, or cooking vessels visible and their state (e.g., 'pan is heating', 'knife is on the cutting board', 'onions are translucent'), (2) Any visual cues about cooking progress or technique (e.g., 'oil shimmering', 'vegetables browning', 'steam rising').",
) -> str:
    """Generate a textual description of a frame using an LLM.
    
    Args:
        frame: PIL Image to describe
        llm: LLM generator instance
        prompt: Prompt for frame description
        
    Returns:
        Textual description of the frame
    """
    # Convert frame to base64 data URL
    frame_data_url = image_to_base64_data_url(frame)
    
    # Create multimodal message with image
    # The content is a list with text and image components
    multimodal_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": frame_data_url}}
    ]
    
    # Pass the multimodal content to the LLM
    # The LLMGenerator should support this format for vision models
    messages = [("user", multimodal_content)]
    
    # Generate description
    try:
        response = llm.generate(messages)
        return response[0].strip()
    
    except Exception as e:
        print(f"Error generating frame description: {e}")
        return "[Frame description unavailable]"


def create_multimodal_message_for_video(
    prompt_text: str,
    video_bytes: bytes,
) -> List[Dict]:
    """Create a multimodal message with text and video for Gemini.
    
    Args:
        prompt_text: Text prompt
        video_bytes: Video file as bytes
        
    Returns:
        Message content list for multimodal input
    """
    # Encode video to base64
    video_b64 = base64.b64encode(video_bytes).decode()
    video_data_url = f"data:video/mp4;base64,{video_b64}"
    
    return [
        {"type": "text", "text": prompt_text},
        {"type": "video_url", "video_url": {"url": video_data_url}}
    ]


def create_multimodal_message_for_frames(
    prompt_text: str,
    frames: List[Image.Image],
) -> List[Dict]:
    """Create a multimodal message with text and multiple frames.
    
    Args:
        prompt_text: Text prompt
        frames: List of PIL Images
        
    Returns:
        Message content list for multimodal input
    """
    content = [{"type": "text", "text": prompt_text}]
    
    for frame in frames:
        frame_data_url = image_to_base64_data_url(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": frame_data_url}
        })
    
    return content
