#!/usr/bin/env python3
"""
Interactive terminal interface for prompting LLaVA with system prompt.
Automatically prepends the system prompt from system_prompt.txt to every user query.
"""

import ollama
import base64
import argparse
import sys
from pathlib import Path


def load_system_prompt(prompt_file='system_prompt.txt'):
    """Load the system prompt from file."""
    prompt_path = Path(__file__).parent / prompt_file
    
    if not prompt_path.exists():
        print(f"Warning: System prompt file not found at {prompt_path}")
        return ""
    
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def encode_image(image_path):
    """Encode image file to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def prompt_llava(user_prompt, image_paths=None, system_prompt="", model='llava', verbose=False):
    """
    Send a prompt to LLaVA with optional images.
    
    Args:
        user_prompt: The user's query
        image_paths: List of image file paths (optional)
        system_prompt: System prompt to prepend
        model: Ollama model name
        verbose: Print full prompt if True
    
    Returns:
        Response text from LLaVA
    """
    # Combine system prompt with user prompt
    full_prompt = system_prompt
    if full_prompt:
        full_prompt += "\n\n" + user_prompt
    else:
        full_prompt = user_prompt
    
    if verbose:
        print(f"\n{'='*80}")
        print("FULL PROMPT SENT TO LLAVA:")
        print(f"{'='*80}")
        print(full_prompt)
        print(f"{'='*80}\n")
    
    # Encode images if provided
    images = []
    if image_paths:
        for img_path in image_paths:
            img_path = Path(img_path)
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue
            images.append(encode_image(img_path))
        
        if verbose and images:
            print(f"Attached {len(images)} image(s): {[str(p) for p in image_paths]}\n")
    
    # Build message
    message = {
        'role': 'user',
        'content': full_prompt
    }
    
    if images:
        message['images'] = images
    
    # Call LLaVA
    try:
        response = ollama.chat(
            model=model,
            messages=[message]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"


def interactive_mode(system_prompt, model='llava'):
    """Run in interactive mode with continuous prompting."""
    print("="*80)
    print("LLaVA Interactive Terminal")
    print("="*80)
    print("System prompt loaded and will be prepended to all queries.")
    print("\nCommands:")
    print("  - Type your prompt and press Enter")
    print("  - Use '/image <path>' to attach an image")
    print("  - Use '/images <path1> <path2> ...' to attach multiple images")
    print("  - Use '/clear' to clear attached images")
    print("  - Use '/show' to show current system prompt")
    print("  - Use '/quit' or Ctrl+C to exit")
    print("="*80)
    
    attached_images = []
    
    while True:
        try:
            # Get user input
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                
                if cmd == '/quit' or cmd == '/exit':
                    print("Goodbye!")
                    break
                
                elif cmd == '/show':
                    print(f"\n{'='*80}")
                    print("CURRENT SYSTEM PROMPT:")
                    print(f"{'='*80}")
                    print(system_prompt if system_prompt else "(none)")
                    print(f"{'='*80}")
                    continue
                
                elif cmd == '/clear':
                    attached_images = []
                    print("Cleared all attached images.")
                    continue
                
                elif cmd == '/image':
                    if len(cmd_parts) < 2:
                        print("Usage: /image <path>")
                        continue
                    img_path = cmd_parts[1].strip()
                    if Path(img_path).exists():
                        attached_images = [img_path]
                        print(f"Attached: {img_path}")
                    else:
                        print(f"Image not found: {img_path}")
                    continue
                
                elif cmd == '/images':
                    if len(cmd_parts) < 2:
                        print("Usage: /images <path1> <path2> ...")
                        continue
                    img_paths = cmd_parts[1].strip().split()
                    valid_paths = [p for p in img_paths if Path(p).exists()]
                    if valid_paths:
                        attached_images = valid_paths
                        print(f"Attached {len(valid_paths)} image(s): {valid_paths}")
                    else:
                        print("No valid images found.")
                    continue
                
                else:
                    print(f"Unknown command: {cmd}")
                    continue
            
            # Process regular prompt
            print("\nProcessing...", end='', flush=True)
            response = prompt_llava(
                user_input,
                image_paths=attached_images if attached_images else None,
                system_prompt=system_prompt,
                model=model
            )
            print("\r" + " "*20 + "\r", end='')  # Clear "Processing..."
            
            print(f"\n{'─'*80}")
            print("RESPONSE:")
            print(f"{'─'*80}")
            print(response)
            print(f"{'─'*80}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description='Interactive terminal interface for LLaVA with system prompt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python prompt_llava.py
  
  # Single prompt without image
  python prompt_llava.py -p "What actions should the robot take?"
  
  # Single prompt with image
  python prompt_llava.py -p "Analyze this scene" -i frame.jpg
  
  # Multiple images
  python prompt_llava.py -p "Compare these frames" -i frame1.jpg frame2.jpg
  
  # Show full prompt being sent
  python prompt_llava.py -p "Describe the scene" -i frame.jpg --verbose
  
  # Skip system prompt
  python prompt_llava.py -p "Simple description" --no-system-prompt
        """
    )
    
    parser.add_argument(
        '-p', '--prompt',
        type=str,
        help='User prompt (if not provided, enters interactive mode)'
    )
    parser.add_argument(
        '-i', '--images',
        type=str,
        nargs='+',
        help='Path(s) to image file(s)'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='llava',
        help='Ollama model name (default: llava)'
    )
    parser.add_argument(
        '--system-prompt-file',
        type=str,
        default='system_prompt.txt',
        help='Path to system prompt file (default: system_prompt.txt)'
    )
    parser.add_argument(
        '--no-system-prompt',
        action='store_true',
        help='Skip prepending system prompt'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show full prompt being sent to LLaVA'
    )
    
    args = parser.parse_args()
    
    # Load system prompt
    system_prompt = ""
    if not args.no_system_prompt:
        system_prompt = load_system_prompt(args.system_prompt_file)
        if not system_prompt:
            print("Warning: System prompt is empty or not found.")
    
    # Single prompt mode
    if args.prompt:
        response = prompt_llava(
            args.prompt,
            image_paths=args.images,
            system_prompt=system_prompt,
            model=args.model,
            verbose=args.verbose
        )
        print(response)
    
    # Interactive mode
    else:
        interactive_mode(system_prompt, model=args.model)


if __name__ == '__main__':
    main()
