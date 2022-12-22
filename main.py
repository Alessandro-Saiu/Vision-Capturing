from transformers import AutoImageProcessor
from transformers import TimesformerForVideoClassification
import torch
from video_processing import VideoReader

VIDEO_PATH = r"C:/Users/aless/Documents/Training/test_256.mp4"

if __name__ == '__main__':
    reader = VideoReader(VIDEO_PATH)
    video = reader.open_video()

    feature_extractor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

    for clip in video:
        inputs = feature_extractor(clip, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class:", model.config.id2label[predicted_class_idx])


