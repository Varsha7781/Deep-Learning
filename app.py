# streamlit_nn_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Neural Network One Iteration Demo", layout="wide")

# ===============================
# ACTIVATION FUNCTIONS
# ===============================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# ===============================
# USER INPUTS
# ===============================
st.sidebar.title("Neural Network Inputs")
x1 = st.sidebar.slider("Input x1", 0.0, 1.0, 0.5)
x2 = st.sidebar.slider("Input x2", 0.0, 1.0, 0.8)

# Input → Hidden Weights
w_x1h1 = st.sidebar.slider("Weight w_x1h1", 0.0, 1.0, 0.4)
w_x2h1 = st.sidebar.slider("Weight w_x2h1", 0.0, 1.0, 0.3)
w_x1h2 = st.sidebar.slider("Weight w_x1h2", 0.0, 1.0, 0.2)
w_x2h2 = st.sidebar.slider("Weight w_x2h2", 0.0, 1.0, 0.7)

# Hidden → Output Weights
w_h1y = st.sidebar.slider("Weight w_h1y", 0.0, 1.0, 0.6)
w_h2y = st.sidebar.slider("Weight w_h2y", 0.0, 1.0, 0.9)

target = st.sidebar.slider("Target Output", 0.0, 1.0, 1.0)
lr = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)

st.title("🔹 Neural Network: One Training Iteration Demo")

# ===============================
# FUNCTION TO DRAW CONNECTIVITY
# ===============================
def draw_network(weights, title="Neural Network Connectivity"):
    input_pos = {'x1': (0,3), 'x2': (0,1)}
    hidden_pos = {'h1': (2,3), 'h2': (2,1)}
    output_pos = {'y': (4,2)}

    fig, ax = plt.subplots(figsize=(8,5))
    ax.axis('off')
    ax.set_xlim(-1,5)
    ax.set_ylim(0,4)

    # Draw nodes
    for name, (x,y) in {**input_pos, **hidden_pos, **output_pos}.items():
        ax.plot(x, y, 'o', markersize=30,
                color='skyblue' if name in input_pos else 'lightgreen' if name in hidden_pos else 'salmon')
        ax.text(x, y, name, fontsize=12, ha='center', va='center')

    # Draw connections
    connections = [
        ('x1','h1', weights['w_x1h1']),
        ('x2','h1', weights['w_x2h1']),
        ('x1','h2', weights['w_x1h2']),
        ('x2','h2', weights['w_x2h2']),
        ('h1','y', weights['w_h1y']),
        ('h2','y', weights['w_h2y'])
    ]
    for src, dst, w in connections:
        x_start, y_start = {**input_pos, **hidden_pos, **output_pos}[src]
        x_end, y_end = {**input_pos, **hidden_pos, **output_pos}[dst]
        ax.annotate("",
                    xy=(x_end, y_end),
                    xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle="->", lw=w*3, color='gray'))
        xm, ym = (x_start + x_end)/2, (y_start + y_end)/2
        ax.text(xm, ym+0.1, f"{w:.2f}", fontsize=10, color='blue', ha='center')

    plt.title(title)
    return fig

# ===============================
# SHOW INITIAL NETWORK
# ===============================
st.subheader("Initial Neural Network")
weights = {
    'w_x1h1': w_x1h1, 'w_x2h1': w_x2h1,
    'w_x1h2': w_x1h2, 'w_x2h2': w_x2h2,
    'w_h1y': w_h1y, 'w_h2y': w_h2y
}
st.pyplot(draw_network(weights, "Initial Neural Network"))

# ===============================
# FORWARD PROPAGATION
# ===============================
z_h1 = x1*w_x1h1 + x2*w_x2h1
z_h2 = x1*w_x1h2 + x2*w_x2h2
h1 = sigmoid(z_h1)
h2 = sigmoid(z_h2)
y = h1*w_h1y + h2*w_h2y
loss = 0.5*(target - y)**2
error = y - target

st.subheader("Forward Propagation Results")
st.write(f"Hidden Net Inputs: z_h1 = {z_h1:.4f}, z_h2 = {z_h2:.4f}")
st.write(f"Hidden Outputs: h1 = {h1:.4f}, h2 = {h2:.4f}")
st.write(f"Predicted Output: y = {y:.4f}, Target = {target}, Error = {error:.4f}, MSE Loss = {loss:.6f}")

# Step 1: Hidden Layer Net Inputs
st.subheader("Step 1: Hidden Layer Net Inputs")
fig, ax = plt.subplots()
ax.bar(["z_h1", "z_h2"], [z_h1, z_h2], color='orange')
ax.set_ylabel("Net Input Value")
st.pyplot(fig)

# Step 2: Sigmoid Activation
st.subheader("Step 2: Sigmoid Activation")
z_vals = np.linspace(-5,5,200)
fig, ax = plt.subplots()
ax.plot(z_vals, sigmoid(z_vals), label="Sigmoid Curve")
ax.scatter([z_h1, z_h2], [h1, h2], color='red', label="Neuron Outputs")
ax.set_xlabel("z")
ax.set_ylabel("σ(z)")
ax.legend()
st.pyplot(fig)

# Step 3: Target vs Predicted Output
st.subheader("Step 3: Target vs Predicted Output")
fig, ax = plt.subplots()
ax.bar(["Target", "Predicted"], [target, y], color=['green','red'])
st.pyplot(fig)

# Step 4: MSE Loss
st.subheader("Step 4: MSE Loss")
fig, ax = plt.subplots()
ax.bar(["MSE Loss"], [loss], color='purple')
st.pyplot(fig)

# ===============================
# BACKPROPAGATION
# ===============================
grad_w_h1y = error * h1
grad_w_h2y = error * h2
delta_h1 = error * w_h1y * sigmoid_derivative(h1)
delta_h2 = error * w_h2y * sigmoid_derivative(h2)

st.subheader("Backpropagation")
st.write(f"Gradients: ∂E/∂w_h1y = {grad_w_h1y:.6f}, ∂E/∂w_h2y = {grad_w_h2y:.6f}")
st.write(f"Hidden Layer Deltas: delta_h1 = {delta_h1:.6f}, delta_h2 = {delta_h2:.6f}")

# Step 5: Output Layer Gradients
st.subheader("Step 5: Output Layer Gradients")
fig, ax = plt.subplots()
ax.bar(["∂E/∂w_h1y","∂E/∂w_h2y"], [grad_w_h1y, grad_w_h2y], color='red')
st.pyplot(fig)

# ===============================
# WEIGHT UPDATES
# ===============================
w_h1y_new = w_h1y - lr*grad_w_h1y
w_h2y_new = w_h2y - lr*grad_w_h2y
w_x1h1_new = w_x1h1 - lr*delta_h1*x1
w_x2h1_new = w_x2h1 - lr*delta_h1*x2
w_x1h2_new = w_x1h2 - lr*delta_h2*x1
w_x2h2_new = w_x2h2 - lr*delta_h2*x2

# Step 6: Weight Updates
st.subheader("Step 6: Updated Hidden → Output Weights")
fig, ax = plt.subplots()
ax.bar(["Old w_h1y","New w_h1y","Old w_h2y","New w_h2y"],
       [w_h1y,w_h1y_new,w_h2y,w_h2y_new],
       color=['blue','green','blue','green'])
st.pyplot(fig)

# ===============================
# SHOW UPDATED NETWORK
# ===============================
st.subheader("Updated Neural Network")
updated_weights = {
    'w_x1h1': w_x1h1_new, 'w_x2h1': w_x2h1_new,
    'w_x1h2': w_x1h2_new, 'w_x2h2': w_x2h2_new,
    'w_h1y': w_h1y_new, 'w_h2y': w_h2y_new
}
st.pyplot(draw_network(updated_weights, "Updated Neural Network"))

# ===============================
# FINAL NUMERICAL OUTPUTS
# ===============================
st.subheader("Final Numerical Results")
st.write(f"h1 = {h1:.4f}, h2 = {h2:.4f}")
st.write(f"Final output y = {y:.4f}, MSE Loss = {loss:.6f}")
st.write("Updated Weights:")
st.write(f"w_x1h1 = {w_x1h1_new:.4f}, w_x2h1 = {w_x2h1_new:.4f}")
st.write(f"w_x1h2 = {w_x1h2_new:.4f}, w_x2h2 = {w_x2h2_new:.4f}")
st.write(f"w_h1y = {w_h1y_new:.4f}, w_h2y = {w_h2y_new:.4f}")

st.success("✅ One Training Iteration Completed Successfully!")
