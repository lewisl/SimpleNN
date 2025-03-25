```python
import numpy as np

def backprop_cnn(layers, learning_rate):
    """
    Backpropagation for CNN with structure:
    l1: input
    l2: conv + relu
    l3: conv + relu
    l4: maxpool
    l5: flatten
    l6: linear + relu
    l7: linear + relu
    l8: linear + softmax
    """
    # Store gradients for each layer
    gradients = {}
    
    # Starting from the last layer (l8), compute gradients backwards
    def compute_l8_gradients(output_grad):
        """Softmax + Linear layer gradients"""
        # Softmax gradient
        softmax_grad = output_grad * (layers['l8'] * (1 - layers['l8']))
        # Linear gradient
        gradients['W8'] = np.dot(layers['l7'].T, softmax_grad)
        gradients['b8'] = np.sum(softmax_grad, axis=0)
        # Gradient for previous layer
        return np.dot(softmax_grad, layers['W8'].T)
    
    def compute_l7_gradients(l8_grad):
        """Linear + ReLU layer gradients"""
        # ReLU gradient
        relu_grad = l8_grad * (layers['l7'] > 0)
        # Linear gradient
        gradients['W7'] = np.dot(layers['l6'].T, relu_grad)
        gradients['b7'] = np.sum(relu_grad, axis=0)
        return np.dot(relu_grad, layers['W7'].T)
    
    def compute_l6_gradients(l7_grad):
        """Linear + ReLU layer gradients"""
        relu_grad = l7_grad * (layers['l6'] > 0)
        gradients['W6'] = np.dot(layers['l5'].T, relu_grad)
        gradients['b6'] = np.sum(relu_grad, axis=0)
        return np.dot(relu_grad, layers['W6'].T)
    
    def compute_l5_gradients(l6_grad):
        """Flatten layer gradient (reshape back to conv shape)"""
        return l6_grad.reshape(layers['l4'].shape)
    
    def compute_l4_gradients(l5_grad):
        """Maxpool layer gradient"""
        pool_grad = np.zeros_like(layers['l3'])
        # For each position where max was selected, pass gradient back
        for i in range(0, pool_grad.shape[2], 2):
            for j in range(0, pool_grad.shape[3], 2):
                window = layers['l3'][:, :, i:i+2, j:j+2]
                mask = window == np.max(window, axis=(2,3), keepdims=True)
                pool_grad[:, :, i:i+2, j:j+2] = mask * l5_grad[:, :, i//2, j//2, None, None]
        return pool_grad
    
    def compute_l3_gradients(l4_grad):
        """Conv + ReLU layer gradients"""
        # ReLU gradient
        relu_grad = l4_grad * (layers['l3'] > 0)
        # Convolution gradient
        gradients['W3'] = np.zeros_like(layers['W3'])
        gradients['b3'] = np.sum(relu_grad, axis=(0,2,3))
        
        # Compute weight gradients through convolution
        for i in range(relu_grad.shape[2]):
            for j in range(relu_grad.shape[3]):
                window = layers['l2'][:, :, i:i+3, j:j+3]
                for k in range(relu_grad.shape[1]):
                    gradients['W3'][:, k] += np.sum(window * relu_grad[:, k, i, j][:, None, None, None], axis=0)
        
        # Compute gradient for previous layer through full convolution
        pad_grad = np.pad(relu_grad, ((0,0), (0,0), (1,1), (1,1)))
        prev_grad = np.zeros_like(layers['l2'])
        for i in range(prev_grad.shape[2]):
            for j in range(prev_grad.shape[3]):
                window = pad_grad[:, :, i:i+3, j:j+3]
                prev_grad[:, :, i, j] = np.sum(window * layers['W3'][:, :, None, None], axis=(1,2,3))
        return prev_grad
    
    def compute_l2_gradients(l3_grad):
        """Conv + ReLU layer gradients"""
        # ReLU gradient
        relu_grad = l3_grad * (layers['l2'] > 0)
        # Convolution gradient
        gradients['W2'] = np.zeros_like(layers['W2'])
        gradients['b2'] = np.sum(relu_grad, axis=(0,2,3))
        
        # Compute weight gradients through convolution
        for i in range(relu_grad.shape[2]):
            for j in range(relu_grad.shape[3]):
                window = layers['l1'][:, :, i:i+3, j:j+3]
                for k in range(relu_grad.shape[1]):
                    gradients['W2'][:, k] += np.sum(window * relu_grad[:, k, i, j][:, None, None, None], axis=0)
        
        # Compute gradient for input layer through full convolution
        pad_grad = np.pad(relu_grad, ((0,0), (0,0), (1,1), (1,1)))
        input_grad = np.zeros_like(layers['l1'])
        for i in range(input_grad.shape[2]):
            for j in range(input_grad.shape[3]):
                window = pad_grad[:, :, i:i+3, j:j+3]
                input_grad[:, :, i, j] = np.sum(window * layers['W2'][:, :, None, None], axis=(1,2,3))
        return input_grad
    
    # Chain the gradients backwards
    def backward_pass(output_grad):
        l8_grad = compute_l8_gradients(output_grad)
        l7_grad = compute_l7_gradients(l8_grad)
        l6_grad = compute_l6_gradients(l7_grad)
        l5_grad = compute_l5_gradients(l6_grad)
        l4_grad = compute_l4_gradients(l5_grad)
        l3_grad = compute_l3_gradients(l4_grad)
        l2_grad = compute_l2_gradients(l3_grad)
        
        # Update weights using gradients
        for layer in [2, 3, 6, 7, 8]:
            layers[f'W{layer}'] -= learning_rate * gradients[f'W{layer}']
            layers[f'b{layer}'] -= learning_rate * gradients[f'b{layer}']
        
        return gradients
    
    return backward_pass

```

I've implemented the backpropagation algorithm for your CNN architecture. Let me explain the key aspects:

1. Layer Structure:
   - Each layer's gradients are computed in separate functions for clarity
   - The notation follows your request: l for current layer, l+1 for next layer in backprop

2. Key Components:
   - For convolutional layers (l2, l3):
     - Handles both ReLU gradient and convolution gradient
     - Computes gradients for weights, biases, and previous layer
   - For max pooling (l4):
     - Creates a mask to route gradients through max positions
   - For fully connected layers (l6, l7):
     - Combines linear and ReLU gradients
   - For final layer (l8):
     - Handles softmax gradient calculation

3. Gradient Flow:
   - The backward_pass function chains all gradients together
   - Gradients flow from loss through l8 down to l2
   - Weight updates are applied using the learning rate

Would you like me to explain any specific part in more detail? I can elaborate on:
- The convolution gradient computation
- The maxpool backprop implementation
- The ReLU gradient calculation
- The weight update process