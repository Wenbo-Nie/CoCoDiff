import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

def load_and_preprocess_image(image_path: str, image_size: int = 256) -> torch.Tensor:

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(image)

def check_image_quality_optimized(content_image: torch.Tensor, 
                                  style_image: torch.Tensor, 
                                  generated_image: torch.Tensor,
                                  sobel_threshold: float, 
                                  gram_threshold: float) -> bool:
    content_image = torch.clamp(content_image, 0, 1)
    style_image = torch.clamp(style_image, 0, 1)
    generated_image = torch.clamp(generated_image, 0, 1)

    content_gray = transforms.functional.rgb_to_grayscale(content_image)
    generated_gray = transforms.functional.rgb_to_grayscale(generated_image)

    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    if generated_gray.device != sobel_kernel_x.device:
        sobel_kernel_x = sobel_kernel_x.to(generated_gray.device)
        sobel_kernel_y = sobel_kernel_y.to(generated_gray.device)

    sobel_x_content = F.conv2d(content_gray.unsqueeze(0), sobel_kernel_x, padding=1)
    sobel_y_content = F.conv2d(content_gray.unsqueeze(0), sobel_kernel_y, padding=1)
    sobel_magnitude_content = torch.sqrt(sobel_x_content**2 + sobel_y_content**2)

    sobel_x_generated = F.conv2d(generated_gray.unsqueeze(0), sobel_kernel_x, padding=1)
    sobel_y_generated = F.conv2d(generated_gray.unsqueeze(0), sobel_kernel_y, padding=1)
    sobel_magnitude_generated = torch.sqrt(sobel_x_generated**2 + sobel_y_generated**2)
    
    sobel_loss = F.l1_loss(sobel_magnitude_generated, sobel_magnitude_content)

    def gram_matrix(input_tensor):
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        a, b, c, d = input_tensor.size()
        features = input_tensor.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    gram_style = gram_matrix(style_image)
    gram_generated = gram_matrix(generated_image)
    
    gram_loss = F.l1_loss(gram_generated, gram_style)

    print(f"Sobel Loss: {sobel_loss.item():.4f}, Gram Loss: {gram_loss.item():.4f}")
    
    if sobel_loss.item() > sobel_threshold or gram_loss.item() < gram_threshold:
        print("Conditions not met: Either content is distorted or style is not captured well.")
        return False
    else:
        print("Conditions met: Content is well-preserved and style is well-captured.")
        return True

if __name__ == '__main__':
    
    content_path = ""
    style_path = ""
    generated_path = ""

    try:
        content_img = load_and_preprocess_image(content_path)
        style_img = load_and_preprocess_image(style_path)
        generated_img = load_and_preprocess_image(generated_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your image paths.")
        exit()

  
    SOBEL_THRESHOLD = 
    GRAM_THRESHOLD = 
    
    result = check_image_quality_optimized(content_img, style_img, generated_img, SOBEL_THRESHOLD, GRAM_THRESHOLD)
    print(f"Final Result: {result}")
