# Models

Place your HANNAH `.pt` checkpoint file in this folder.

The model weights are not included in this repository due to their large size (~1.4 GB). You can download them from our private HuggingFace repository:

**[HannahTeam/Proyecto-Hannah-360M](https://huggingface.co/HannahTeam/Proyecto-Hannah-360M)**

## Instructions

1. Download the `.pt` file from HuggingFace (e.g. `hannah_personality_final.pt`)
2. Place it in this `models/` folder
3. Start the server with `python app.py`

The server will automatically detect the most recent `.pt` file in this folder.
