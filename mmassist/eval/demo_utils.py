import os
import copy
import imageio
import textwrap
import numpy as np
from PIL import ImageFont, ImageDraw

FONT_FILE = os.path.join(os.path.dirname(__file__), "../../assets/Consolas.ttf")
FONT = ImageFont.truetype(FONT_FILE, 14)
FONT_BIG = ImageFont.truetype(FONT_FILE, 18)


def wrap_text(text: str, width: int) -> str:
    wrapped_text = textwrap.wrap(text, width=width)
    return "\n".join(wrapped_text)


def draw_text(
    draw,
    text,
    x=0,
    y=0,
    margin: int = 5,
    box_color=(128, 0, 0, 128),
    text_color=(255, 255, 255),
    font=FONT,
    max_num_char_per_line=48,  # good for 384px width, 14pt font
    img_width=384,
):

    if max_num_char_per_line > 0:
        text = wrap_text(text, max_num_char_per_line)
    _, _, txt_width, txt_height = draw.textbbox((0, 0), text=text, font=font)

    x1_box, y1_box = x, y
    # box_w, box_h = txt_width + margin * 2, txt_height + margin * 2
    box_w, box_h = img_width, txt_height + margin * 2
    x2_box, y2_box = x1_box + box_w, y1_box + box_h
    draw.rectangle([x1_box, y1_box, x2_box, y2_box], fill=box_color)

    x_txt, y_txt = x + margin, y + margin
    draw.text((x_txt, y_txt), text, fill=text_color, font=font)

    return box_w, box_h


def annotate_and_save_video(
    streams: list[dict],
    output_file: str,
    fps: int = 2,
    show: str | None = "time",
    assistant_name_gt: str = "REF",
    assistant_name_gen: str = "GEN",
    pause_time: int | str = "adaptive",
    add_system_prompt: bool = True,
) -> list[dict]:
    writer = imageio.get_writer(output_file, fps=fps)
    frame_idx_in_video = 0
    for frame_idx, frame in enumerate(streams):
        time = frame["timestamp_in_stream"]
        text_inputs = frame["text_inputs"]
        gen = frame["gen"]
        ref = frame["ref"]
        img = frame["image"]
        if img is None:
            print(f"Failed to create video as image is None at frame {frame_idx}")
            return None

        img = copy.deepcopy(img)
        draw = ImageDraw.Draw(img, "RGBA")

        # Draw the time
        if show == "time":
            txt_draw = f"{time:.1f}s"
        elif show == "index":
            txt_draw = f"Frame {frame_idx}"
        else:
            txt_draw = ""
        if txt_draw:
            _, _, txt_width, txt_height = draw.textbbox((0, 0), text=txt_draw, font=FONT_BIG)
            tx = img.width - txt_width - 5
            ty = img.height - txt_height - 10 #- 70
            draw_text(draw, txt_draw, x=tx, y=ty, font=FONT_BIG, box_color=(0, 100, 0, 128))

        # Draw the input text if any
        y = 0
        input_text = ""
        if text_inputs:
            for role, text in text_inputs:
                if not add_system_prompt and role == "system":
                    continue
                color = (128, 0, 0, 128) if role == "user" else (0, 100, 0, 128)
                _, h = draw_text(draw, f"{role.upper()}: {text}", y=y, box_color=color)
                y += h + 1
                input_text += f"{text} "

        # Draw the reference assistant utterance
        if ref:
            msg = f"{assistant_name_gt}: {ref}"
            _, h = draw_text(draw, msg, y=y, box_color=(0, 0, 128, 128))
            y += h + 1

        # Draw the generated assistant utterance
        if gen:
            msg = f"{assistant_name_gen}: {gen}"
            _, h = draw_text(draw, msg, y=y, box_color=(0, 0, 128, 128))

        # pause when there is some output
        if input_text or ref or gen:
            total_words = len(f"{input_text} {ref} {gen}".split())
            if isinstance(pause_time, str):
                pause_time = min(total_words / 300 * 60, 10)  # 300 words/min, up to 10s
            else:
                pause_time = pause_time
            num_repeat = int(pause_time * fps)
        else:
            num_repeat = 1

        data = np.array(img)
        frame["frame_idx_in_video"] = frame_idx_in_video
        for _ in range(num_repeat):
            writer.append_data(data)
            frame_idx_in_video += 1

    # Close the writer and reader
    writer.close()

    return streams