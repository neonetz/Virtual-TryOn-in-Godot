# Godot Frontend for Virtual Try-On

This directory contains the Godot 4 project that acts as the frontend for the Virtual Try-On application. It receives a video stream from the Python backend and provides a user interface for interacting with the application.

## Scenes

The project is organized into several key scenes:

-   **`scenes/menus/MainMenu.tscn`**: The main entry point of the application. It displays the main menu with options to start the try-on, view settings, see the about page, or quit.
-   **`scenes/tryon/TryOn.tscn`**: The core scene for the virtual try-on experience. It displays the video feed from the backend and includes the UI for mask selection.
-   **`scenes/menus/About.tscn`**: A simple scene that displays information about the project and its creators.
-   **`scenes/menus/Settings.tscn`**: A placeholder for any future application settings.

## Scripts

The functionality of the Godot application is driven by a few key GDScript files:

-   **`scripts/tryon/udp_video_receiver.gd`**: This is the most important script for communication. It creates a UDP socket to listen for incoming video frames from the Python backend. When a packet is received, it parses the JSON data, decodes the base64 JPEG image, and updates a `TextureRect` to display the frame. It also handles connection timeouts and displays statistics like FPS and the number of detected faces.

-   **`scripts/tryon/mask_scroll_selector.gd`**: This script dynamically creates the mask selection UI. It scans the `assets/masks` directory for all available mask images (`.png` files) and creates a button for each one. When a mask button is clicked, it sends a UDP command to the backend on a separate port (`5556`) to instruct it to change the active mask.

-   **`scripts/ui/main_menu.gd`**: This script handles the logic for the main menu, including navigating to other scenes and playing sound effects.

-   **`scripts/tryon/tryon.gd`**: This script manages the main "Try On" scene, primarily handling the "Back" button functionality to return to the main menu.

## How It Works

1.  **Initialization**: When the `TryOn.tscn` scene is loaded, the `udp_video_receiver.gd` script initializes a `PacketPeerUDP` and binds it to port `5555` to start listening for data.

2.  **Receiving Data**: The script continuously checks for available packets in its `_process` loop. When a packet arrives, it's parsed as a JSON string.

3.  **Image Decoding**: The `image` field from the JSON data, which is a base64 encoded string of a JPEG image, is decoded into a raw byte array. This byte array is then loaded into an `Image` object.

4.  **Display**: An `ImageTexture` is created from the `Image` object, and this texture is assigned to the `texture` property of a `TextureRect` node in the scene. This process is repeated for every frame received, creating the illusion of a real-time video stream.

5.  **Sending Commands**: The `mask_scroll_selector.gd` script also uses a `PacketPeerUDP` to send data. When a user clicks a mask, it constructs a JSON command like `{"command": "change_mask", "mask": "mask-2.png"}` and sends it to the backend on port `5556`. The Python backend listens for these commands and updates its internal state accordingly.

## Assets

-   **`assets/masks/`**: Contains the `.png` images for the masks that can be overlaid on the user's face. The `mask_scroll_selector.gd` script reads directly from this directory.
-   **`assets/backgrounds/`**: Background images used in the UI.
-   **`assets/fonts/`**: Custom fonts used for the UI text.
-   **`assets/sounds/`**: Sound effects for button clicks, hovers, and background ambiance.
-   **`assets/ui/`**: UI elements like button textures.
