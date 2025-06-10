

import os
import ssl
import torch
import logging
import together
import gradio as gr
from PIL import Image
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

ssl._create_default_https_context = ssl._create_unverified_context
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
os.environ["TOGETHER_API_KEY"] = "tgp_v1_wpMDTCEa-9VJLlcMt_H0fZfhisxI_ebnLqVmBFhaMXs"
together.api_key = os.environ["TOGETHER_API_KEY"]
logging.info("Loading MobileNetV2 model...")
weights = MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.eval()
logging.info("Model loaded successfully.")
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def identify_waste_item(image: Image.Image) -> str:
    """
    Identifies the class label of the waste item in the image using MobileNetV2.
    
    Args:
        image (PIL.Image.Image): Input waste item image.
        
    Returns:
        str: Predicted label/category of the waste item.
    """
    try:
        tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(tensor)
        idx = torch.argmax(output, dim=1).item()
        label = weights.meta["categories"][idx]
        logging.info(f"Predicted label: {label}")
        return label
    except Exception as e:
        logging.error(f"Error in identifying image: {e}")
        return "Unknown Item"


def generate_reuse_ideas(label: str) -> str:
    """
    Generates 5 creative and eco-friendly reuse/recycling ideas using Together AI.
    
    Args:
        label (str): Waste item label predicted by the image model.
        
    Returns:
        str: Step-by-step reuse/recycling instructions.
    """
    prompt = (
        f"Suggest 5 creative and practical reuse or recycling ideas for a '{label}'. "
        "Provide step-by-step instructions for each idea."
    )

    try:
        logging.info("Sending request to Together AI for reuse ideas...")
        response = together.Completion.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
            top_p=0.9,
            stream=False,
        )
        ideas = response.choices[0].text.strip()
        logging.info("Reuse ideas generated successfully.")
        return ideas
    except Exception as e:
        logging.error(f"Failed to generate reuse ideas: {e}")
        return "Error generating reuse ideas. Please try again later."


def classify_and_generate(img: Image.Image) -> str:
    """
    Full pipeline function: Classify waste item and generate reuse suggestions.
    
    Args:
        img (PIL.Image.Image): Uploaded image.
    
    Returns:
        str: Markdown output with label and instructions.
    """
    label = identify_waste_item(img)
    instructions = generate_reuse_ideas(label)
    return f"**Identified Item:** {label}\n\n**Reuse Instructions:**\n{instructions}"


# ----------------------------- GRADIO WEB INTERFACE -----------------------------

interface = gr.Interface(
    fn=classify_and_generate,
    inputs=gr.Image(type="pil", label="Upload Waste Item Image"),
    outputs=gr.Markdown(label="AI-Generated Reuse Instructions"),
    title="♻️ AI-Powered Waste Reuse Generator",
    description=(
        "Upload an image of a waste item (e.g., plastic bottle, cardboard, bag, etc.).\n\n"
        "The system will identify the item and suggest 5 creative reuse/recycling ideas, "
        "with clear step-by-step instructions using AI.\n\n"
        "Powered by MobileNetV2 and Together AI Mixtral-8x7B."
    ),
    examples=[
        ["example_plastic_bottle.jpg"],
        ["example_cardboard_box.jpg"],
    ],
    allow_flagging="never",
    theme="default"
)

# ----------------------------- MAIN ENTRY POINT -----------------------------

if __name__ == "__main__":
    logging.info("Launching Gradio interface...")
    interface.launch(share=True)  # share=True to allow public URL (for demo/presentation)
