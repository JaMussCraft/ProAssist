from dataclasses import dataclass, asdict


@dataclass
class PreparedSample:
    dataset: str
    """The dataset name."""
    video_uid: str
    """The video unique id."""
    clip_idx: int
    """The clip index in the video as each video may be split into multiple clips."""
    frames_file: str
    """Reletive path to the frames file."""
    max_seq_len: int
    """The maximum sequence length of tokens."""
    seq_len: int
    """The actual number of tokens in the sequence."""
    num_tokens_per_img: int
    """The number of tokens per image."""
    use_img_sep_token: bool
    """Whether the separator token is used between frames."""
    start_frame_idx: int
    """The start frame index of the sample in the frames file."""
    end_frame_idx: int
    """The end frame index of the sample in the frames file."""
    conversation: list[dict]
    """The user-assistant conversation."""
    fps: float | None = None
    """The frames per second of the video."""
    metadata: str | None = None
    """Other dataset-specific metadata in the json string format."""

    def to_dict(self):
        return asdict(self)
