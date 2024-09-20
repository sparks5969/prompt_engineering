import gensim.downloader as api
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px

# Load a pre-trained word embedding model
model_name = 'glove-twitter-100'
model = api.load(model_name)

# Select a subset of words to visualize
words = ['cat', 'dog', 'fish', 'bird', 'lion', 'tiger', 'bear', 'wolf', 'fox', 'eagle',
         'happy', 'sad', 'angry', 'excited', 'calm', 'nervous', 'relaxed', 'tired', 'energetic', 'bored']

# Get the word vectors
word_vectors = [model[word] for word in words]

# Reduce dimensionality to 3D using PCA
pca = PCA(n_components=3)
vectors_3d = pca.fit_transform(word_vectors)

# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=vectors_3d[:, 0],
    y=vectors_3d[:, 1],
    z=vectors_3d[:, 2],
    mode='markers+text',
    text=words,
    hoverinfo='text',
    marker=dict(
        size=10,
        color=list(range(len(words))),
        colorscale='Viridis',
        opacity=0.8
    )
)])

# Update the layout
fig.update_layout(
    title=f'3D Visualization of Word Embeddings ({model_name})',
    scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='PCA Component 3'
    ),
    width=900,
    height=700,
)

# Show the plot
fig.show()

# To create an animation, we can rotate the camera
def rotate_camera(fig, n_frames=100):
    frames = []
    for i in range(n_frames):
        frame = go.Frame(
            layout=dict(
                scene=dict(
                    camera=dict(
                        eye=dict(
                            x=np.cos(i/n_frames * 2*np.pi),
                            y=np.sin(i/n_frames * 2*np.pi),
                            z=0.5
                        )
                    )
                )
            )
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 50, "redraw": True},
                                       "fromcurrent": True,
                                       "transition": {"duration": 0}}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}])
                    ])]
    )

# Apply the rotation animation
rotate_camera(fig)

# Show the animated plot
fig.show()
