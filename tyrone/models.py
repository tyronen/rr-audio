import math
import torch
import torch.nn as nn
from torch.nn import functional as F

CNN_MODEL_PATH="data/cnn_model.pth"

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        input_channels=1
        num_classes=10
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Adaptive pooling handles any input size
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


# TODO: experiment with making the PE learnable
# see disection with o3: https://chatgpt.com/share/685a8e42-8f04-8009-b87a-e30b6fbe56b5
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        model_dim: int,
        max_len: int,
        trainable: bool = True,
    ):
        super().__init__()
        if trainable:
            # Learnable positional encoding
            self.pe = nn.Parameter(torch.zeros(1, max_len, model_dim))
            nn.init.trunc_normal_(self.pe, std=0.02)
        else:
            # Fixed sinusoidal positional encoding (original implementation)
            pe = torch.zeros(max_len, model_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, model_dim, 2) * -(math.log(10_000.0) / model_dim))
            broadcast = position * div_term
            pe[:, 0::2] = torch.sin(broadcast)
            pe[:, 1::2] = torch.cos(broadcast)
            pe = pe.unsqueeze(0)  # add batch dimension
            self.register_buffer("pe", pe)

    def forward(self, x):
        # Add dropout to PE for regularization
        pe = self.pe[:, : x.size(1)]
        return x + F.dropout(pe, p=0.1, training=self.training)


class Patchify(nn.Module):
    # think of each patch as an image token (i.e. as a word, if this was NLP)
    def __init__(self, patch_size: int, model_dim: int):
        super().__init__()
        # use conv2d to unfold each image into patches (more efficient on GPU)
        self.proj = nn.Conv2d(1, model_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        # optionally normalise patch embeddings before they enter the transformer proper
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2)
        x = x.permute(0, 2, 1)
        return self.norm(x)


def attention(k_dim, q, k, v, mask_tensor):
    kt = k.transpose(-2, -1)
    # do attention(Q, K, V) = softmax(Q·K^T / sqrt(dims))·V to get hidden state (where · is dot product)
    attn_dot_product = torch.matmul(q, kt)
    attn_scaled = attn_dot_product / math.sqrt(k_dim)
    if mask_tensor is not None:
        attn_scaled = attn_scaled.masked_fill(mask_tensor, -torch.inf)
    attn_probs = torch.softmax(attn_scaled, dim=-1)
    return torch.matmul(attn_probs, v)


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, mask: bool, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.k_dim = model_dim // num_heads
        self.wqkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.endmulti = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.mask = mask

    def rearrange(self, vector, B, L):
        return vector.reshape(B, L, self.num_heads, self.k_dim).transpose(1, 2)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.split(self.model_dim, dim=-1)
        qh = self.rearrange(q, B, L)
        kh = self.rearrange(k, B, L)
        vh = self.rearrange(v, B, L)

        mask_tensor = None
        if self.mask:
            mask_tensor = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        attended = attention(self.k_dim, qh, kh, vh, mask_tensor=mask_tensor)
        concatted = attended.transpose(1, 2).reshape(B, L, self.model_dim)
        concatted = self.dropout(concatted)
        return self.endmulti(concatted)


class CrossAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.k_dim = model_dim // num_heads
        self.wq = nn.Linear(model_dim, model_dim, bias=False)
        self.wk = nn.Linear(model_dim, model_dim, bias=False)
        self.wv = nn.Linear(model_dim, model_dim, bias=False)
        self.endmulti = nn.Linear(model_dim, model_dim, bias=False)

    def rearrange(self, vector, B, L):
        return vector.reshape(B, L, self.num_heads, self.k_dim).transpose(1, 2)

    def forward(self, images, texts):
        B_image, L_image, D_image = images.shape
        B_text, L_text, D_text = texts.shape
        q = self.wq(texts)
        k = self.wk(images)
        v = self.wv(images)
        qh = self.rearrange(q, B_text, L_text)
        kh = self.rearrange(k, B_image, L_image)
        vh = self.rearrange(v, B_image, L_image)
        attended = attention(self.k_dim, qh, kh, vh, mask_tensor=None)
        concatted = attended.transpose(1, 2).reshape(B_text, L_text, self.model_dim)
        return self.endmulti(concatted)


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(model_dim, ffn_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim, bias=True),
        )

    def forward(self, x):
        return self.sequence(x)


class Encoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        # here, 'multi-head dot-product self attention blocks [...] completely replace convolutions' (see 16x16)
        self.mha = SelfAttention(
            model_dim=model_dim, num_heads=num_heads, mask=False, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, ffn_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mhead = self.mha(x)
        addnormed = self.norm1(x + self.dropout(mhead))

        # pass attention output through feed-forward sub-layer (basic MLP)
        ffned = self.ffn(addnormed)
        return self.norm2(addnormed + self.dropout(ffned))


class Decoder(nn.Module):
    def __init__(self, model_dim: int, ffn_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.masked_self_mha = SelfAttention(model_dim=model_dim, num_heads=num_heads, mask=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.cross_mha = CrossAttention(model_dim=model_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim=model_dim, ffn_dim=ffn_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images, text):
        stage1 = self.masked_self_mha(text)
        addnormed_text = self.norm1(text + self.dropout(stage1))
        stage2 = self.cross_mha(images, addnormed_text)
        addnormed_stage2 = self.norm2(addnormed_text + self.dropout(stage2))
        ffned = self.ffn(addnormed_stage2)
        return self.norm3(addnormed_stage2 + self.dropout(ffned))


class BaseTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        model_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_encoders: int,
        max_pe_len: int,
        use_cls: bool,
        dropout: float,
    ):
        super().__init__()
        self.patchify = Patchify(patch_size, model_dim)
        self.use_cls = use_cls
        self.pe = PositionalEncoding(model_dim, max_pe_len)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        def make_encoder() -> nn.Module:
            return Encoder(
                model_dim=model_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )

        self.encoder_series = nn.ModuleList([make_encoder() for _ in range(num_encoders)])

    def forward(self, x):
        B = x.size(0)
        patched = self.patchify(x)
        D = patched.size(-1)
        out = self.pe(patched)
        if self.use_cls:
            cls_expanded = self.cls_token.expand(B, 1, D)
            out = torch.cat([cls_expanded, out], dim=1)
        for encoder in self.encoder_series:
            out = encoder(out)
        return out



class ComplexTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        model_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_coders: int,
        dropout: float,
        train_pe: bool,
    ):
        super().__init__()
        self.base_transformer = BaseTransformer(
            patch_size=patch_size,
            model_dim=model_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_encoders=num_coders,
            dropout=dropout,
            use_cls=False,
            max_pe_len=64,
        )
        self.embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=model_dim)
        self.pe = torch.nn.Embedding(5, model_dim)
        self.register_buffer("rng", torch.arange(5))

        def make_decoder() -> nn.Module:
            return Decoder(
                model_dim=model_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        self.decoder_series = nn.ModuleList([make_decoder() for _ in range(num_coders)])

        self.linear = nn.Linear(model_dim, VOCAB_SIZE)

    def forward(self, images, input_seqs):
        encoded = self.base_transformer(images)
        text = self.embedding(input_seqs)
        text = text + self.pe(self.rng[: text.size(1)])  # type: ignore
        for decoder in self.decoder_series:
            text = decoder(encoded, text)
        return self.linear(text)
