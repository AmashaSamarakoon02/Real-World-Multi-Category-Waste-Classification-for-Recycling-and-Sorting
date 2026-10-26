import os

import torch
import torch.nn.functional as F

from PIL import Image

import config
from dataset import get_clean_transforms
from models import build_model
from scorecam import ScoreCAM, overlay_heatmap


class WasteClassifier:
    def __init__(self, backbone_name):
        self.backbone_name = backbone_name
        self.device = config.DEVICE

        ckpt_path = os.path.join(
            config.CHECKPOINT_DIR,
            f"{backbone_name}_trashnet_best.pt",
        )

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found:\n{ckpt_path}"
            )

        self.model, self.target_layer = build_model(
            backbone_name,
            len(config.TRASHNET_CLASSES),
            pretrained=False,
        )

        self.model.load_state_dict(
            torch.load(
                ckpt_path,
                map_location=self.device,
            )
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        _, self.eval_tf = get_clean_transforms()

        self.scorecam = ScoreCAM(
            self.model,
            self.target_layer,
            self.device,
        )

    @torch.no_grad()
    def predict(self, pil_image):
        tensor = self.eval_tf(
            pil_image
        ).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]

        top_idx = int(probs.argmax())
        confidence = float(probs[top_idx])

        ranked = sorted(
            zip(
                config.TRASHNET_CLASSES,
                probs.tolist(),
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "predicted_class": config.TRASHNET_CLASSES[top_idx],
            "confidence": confidence,
            "all_probabilities": ranked,
        }

    def format_prediction_summary(self, result):
        top_classes = result["all_probabilities"][:3]
        lines = [
            "Prediction summary",
            f"Predicted class: {result['predicted_class']}",
            f"Confidence: {result['confidence'] * 100:.2f}%",
            "Top-3 classes:",
        ]

        for index, (cls, prob) in enumerate(top_classes, start=1):
            lines.append(f"{index}. {cls}: {prob * 100:.2f}%")

        return "\n".join(lines)

    def save_prediction_report(self, result, out_path=None):
        if out_path is None:
            out_path = os.path.join(
                config.RESULTS_DIR,
                "latest_prediction.txt",
            )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(self.format_prediction_summary(result) + "\n")

        return out_path

    def predict_with_explanation(self, pil_image, out_path):
        tensor = self.eval_tf(
            pil_image
        ).unsqueeze(0)

        cam, class_idx = self.scorecam.generate(
            tensor
        )

        overlay = overlay_heatmap(
            pil_image.resize(
                (
                    config.IMG_SIZE,
                    config.IMG_SIZE,
                )
            ),
            cam,
        )

        import matplotlib.pyplot as plt

        plt.imsave(out_path, overlay)

        return (
            config.TRASHNET_CLASSES[class_idx],
            out_path,
        )
