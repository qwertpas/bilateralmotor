from PIL import Image

def compress_gif(input_path, output_path, quality=50, num_colors=128, n=1):
    # Open the original GIF
    img = Image.open(input_path)
    
    # Create a list to hold selected frames
    frames = []
    
    # Iterate through each frame in the GIF
    try:
        frame_count = 0
        while True:
            if frame_count % n == 0:  # Check if the current frame is to be included
                # Convert to P mode which uses a palette of colors
                frame = img.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
                frames.append(frame.copy())  # Append a copy of the frame
            img.seek(frame_count)  # Move to the next frame
            frame_count += 1
    except EOFError:
        pass  # End of frames

    # Save selected frames as a new GIF
    if frames:  # Ensure there are frames to save
        frames[0].save(output_path, save_all=True, append_images=frames[1:], optimize=True, quality=quality, duration=50, loop=0)

# Example usage
compress_gif('motorsim.gif', 'motor_sim_compressed.gif', quality=10, num_colors=16, n=2) 