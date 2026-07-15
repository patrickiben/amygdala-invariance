"""Movie-arm feature extractor -> feat_avg_cache.npz {E,VT,RN} + feat_cache.npz {LL,RC}.

=============================== PROVENANCE NOTE ================================
REFERENCE RE-IMPLEMENTATION. The original extractor that produced the committed
feat_avg_cache.npz / feat_cache.npz was lost (it lived in an ephemeral working
dir). This script was reconstructed from (a) the Methods section, (b) the reader
contract in pls20.py / pls20_ctrl.py / shifted_null.py / combined_controls.py
(which np.load these caches with keys E/VT/RN and LL/RC), (c) the sibling
extractor real_data/scripts/extract_toisoul.py (identical frame->TR-average
convention), and (d) amyg_inv/encoders/{emonet,lowlevel,random_cnn}.py, which
document each encoder's intended real-run form.

It reproduces the PIPELINE and should reproduce the VERDICTS (signs, significance,
the EmoNet-vs-object null, the TOST bound). It is NOT guaranteed to reproduce the
committed 4th-decimal values, because the original run's exact EmoNet preprocessing
and the random-net weights/seed were not preserved. VALIDATE by running this, then
running pls20.py, and confirming the verdict matches real_data/results/pls20_result.txt.
Anywhere marked `# CONFIRM:` is a choice a re-runner should check against Methods.
===============================================================================

Output contract (both caches are TR-averaged, PRE-HRF and PRE-PAL; pls20.prep()
applies the 1.042 PAL stretch, HRF convolution, and PCA at analysis time):
  feat_avg_cache.npz:  E  (EmoNet, Kragel 2019, fc7 4096-d)
                       VT (torchvision ViT-B/16, 768-d CLS)
                       RN (torchvision ResNet-50, 2048-d avgpool)
  feat_cache.npz:      LL (low-level: Gabor energy + luminance + motion)
                       RC (random-weight AlexNet, penultimate 4096-d, fixed seed)
Each array is [n_TR, dim] where n_TR = n_frames // FPS (FPS frames averaged per TR).

Env: SP (cache dir; also where the caches are written), FILM (path to the
legally-obtained film; never redistributed). Run before any movie-arm script.
"""
import os, sys, glob, time, subprocess
import numpy as np

SP = os.environ["SP"]
FILM = os.environ.get("FILM", "")
FPS = 6                      # frames averaged per TR (TR=1s); matches extract_toisoul.py
FRAME_PX = 256              # square frames, as extracted by ffmpeg
BATCH = 64
t0 = time.time()
def log(*a): print(*a, flush=True)


# ----------------------------- frames (shared) ------------------------------
def ensure_frames():
    fdir = f"{SP}/frames6"
    if os.path.isdir(fdir) and len(glob.glob(f"{fdir}/f*.png")) >= 1000:
        return sorted(glob.glob(f"{fdir}/f*.png"))
    os.makedirs(fdir, exist_ok=True)
    assert os.path.exists(FILM), "need FILM=/path/to/500DaysOfSummer.mp4 (copyright-gated, author-supplied)"
    log(f"ffmpeg {FPS}fps {FRAME_PX}x{FRAME_PX} ...")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", FILM,
                    "-vf", f"fps={FPS},scale={FRAME_PX}:{FRAME_PX}", f"{fdir}/f%06d.png"], check=True)
    return sorted(glob.glob(f"{fdir}/f*.png"))


def tr_average(F):
    """[n_frames, dim] -> [n_frames//FPS, dim] by averaging FPS frames per TR."""
    n = (len(F) // FPS) * FPS
    return F[:n].reshape(-1, FPS, F.shape[1]).mean(1).astype(np.float32)


# ------------------------- torch deep encoders ------------------------------
def deep_features(frames):
    """Return dict E/VT/RN/RC as [n_frames, dim] float32 (pre-TR-average)."""
    import torch, cv2
    from torchvision import models, transforms
    DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"deep encoders on device={DEV}")

    # --- EmoNet (Kragel 2019): clone ecco-laboratory/emonet-pytorch, load OSF `amdju` weights,
    #     hook the 4096-d fc7-equivalent penultimate (see amyg_inv/encoders/emonet.py). ---
    emonet_repo = os.environ.get("EMONET_REPO", f"{SP}/emonet_pytorch")
    emonet_wts = os.environ.get("EMONET_WEIGHTS", f"{SP}/emonet/emonet.pth")
    sys.path.insert(0, emonet_repo)
    from emonet_pytorch import EmoNet  # CONFIRM: import path per the ecco-laboratory repo
    emonet = EmoNet()
    emonet.load_state_dict(torch.load(emonet_wts, map_location="cpu"), strict=False)
    emonet.eval().to(DEV)
    ebuf = {}
    # CONFIRM: fc7 is the penultimate 4096-d linear in the AlexNet-based EmoNet classifier.
    emonet.classifier[-2].register_forward_hook(
        lambda m, i, o: ebuf.__setitem__("f", o.flatten(1).detach().cpu().numpy()))

    # --- torchvision object encoders (ImageNet pretrained) + random adversary ---
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1); vit.heads = torch.nn.Identity()
    rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); rn.fc = torch.nn.Identity()
    torch.manual_seed(0)  # CONFIRM: fixed seed for the random-weight adversary (random_cnn.py: "prereg seed")
    rc = models.alexnet(weights=None); rc.classifier = torch.nn.Sequential(*list(rc.classifier.children())[:-1])  # -> 4096-d
    for m in (vit, rn, rc): m.eval().to(DEV)

    # EmoNet uses its own preprocessing; the ImageNet nets use the standard transform.
    imagenet = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    def to_batch(paths, size):
        ims = []
        for fp in paths:
            im = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
            if size != FRAME_PX:
                im = cv2.resize(im, (size, size))
            ims.append(im)
        return torch.from_numpy(np.stack(ims)).float().permute(0, 3, 1, 2).to(DEV) / 255.0

    E, VT, RN, RC = [], [], [], []
    for i in range(0, len(frames), BATCH):
        chunk = frames[i:i + BATCH]
        with torch.no_grad():
            xin = imagenet(to_batch(chunk, 224))
            VT.append(vit(xin).detach().cpu().numpy())
            RN.append(rn(xin).detach().cpu().numpy())
            RC.append(rc(xin).detach().cpu().numpy())
            _ = emonet(to_batch(chunk, 227))  # CONFIRM: EmoNet input size/preproc per emonet-pytorch
            E.append(ebuf["f"].copy())
        if i % (BATCH * 50) == 0:
            log(f"  {i}/{len(frames)} ({time.time()-t0:.0f}s)")
    return {k: np.vstack(v).astype(np.float32) for k, v in
            {"E": E, "VT": VT, "RN": RN, "RC": RC}.items()}


# ------------------------- low-level (no torch) -----------------------------
def lowlevel_features(frames):
    """Gabor energy (4 orient x 2 scale) + mean luminance + frame-to-frame motion.
    Mirrors amyg_inv/encoders/lowlevel.py (GIST-like, no semantic supervision)."""
    import cv2
    from scipy.signal import fftconvolve
    def gabor_bank(n_orient=4, n_scale=2, ksize=15):
        bank = []
        for s in range(n_scale):
            sigma = 2.0 * (s + 1); lam = 4.0 * (s + 1)
            for o in range(n_orient):
                theta = np.pi * o / n_orient
                bank.append(cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, 0.5, 0, ktype=cv2.CV_32F))
        return bank
    bank = gabor_bank()
    feats, prev = [], None
    for k, fp in enumerate(frames):
        g = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        row = [np.sqrt(np.mean(np.maximum(fftconvolve(g, kf, mode="same"), 0.0) ** 2)) for kf in bank]
        row.append(float(g.mean()))                                   # luminance
        row.append(0.0 if prev is None else float(np.mean(np.abs(g - prev))))  # motion energy
        feats.append(row); prev = g
        if k % (BATCH * 50) == 0: log(f"  lowlevel {k}/{len(frames)} ({time.time()-t0:.0f}s)")
    return np.asarray(feats, dtype=np.float32)


def main():
    frames = ensure_frames()
    log(f"{len(frames)} frames @ {FPS}fps -> {len(frames)//FPS} TRs")
    deep = deep_features(frames)
    LL = tr_average(lowlevel_features(frames))
    E, VT, RN, RC = (tr_average(deep[k]) for k in ("E", "VT", "RN", "RC"))
    np.savez(f"{SP}/feat_avg_cache.npz", E=E, VT=VT, RN=RN)
    np.savez(f"{SP}/feat_cache.npz", LL=LL, RC=RC)
    log(f"DONE feat_avg_cache.npz E{E.shape} VT{VT.shape} RN{RN.shape} | "
        f"feat_cache.npz LL{LL.shape} RC{RC.shape} ({time.time()-t0:.0f}s)")
    log("REMINDER: reference re-implementation -- validate pls20.py output against "
        "real_data/results/pls20_result.txt (verdict, not 4th-decimals).")


if __name__ == "__main__":
    main()
