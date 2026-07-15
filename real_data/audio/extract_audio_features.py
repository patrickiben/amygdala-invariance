import os, time, numpy as np, wave, torch
t0=time.time(); DEV="mps" if torch.backends.mps.is_available() else "cpu"
A=os.environ.get("AUDIO_DIR", os.path.join(os.environ.get("AMYG_DATA", os.path.expanduser("~/amyg_data")), "audio"))
from transformers import Wav2Vec2Model, AutoFeatureExtractor
SR=16000
w=wave.open(f"{A}/film_16k.wav"); N=w.getnframes(); aud=np.frombuffer(w.readframes(N),dtype=np.int16).astype(np.float32)/32768.
nsec=N//SR; print("audio sec:",nsec,f"({time.time()-t0:.0f}s)",flush=True)
# ---- audio-affect: emotion-fine-tuned wav2vec2 backbone, pooled per second ----
fe=AutoFeatureExtractor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
m=Wav2Vec2Model.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim").eval().to(DEV)
CH=20  # sec per chunk
emb=[]
for s in range(0,nsec,CH):
    seg=aud[s*SR:min((s+CH),nsec)*SR]
    x=fe(seg,sampling_rate=SR,return_tensors="pt").input_values.to(DEV)
    with torch.no_grad(): h=m(x).last_hidden_state[0].cpu().numpy()  # (frames,1024) ~50Hz
    secs=min(CH,nsec-s); fps=h.shape[0]/max(secs,1)
    for k in range(secs):
        a=int(k*fps); b=int((k+1)*fps); emb.append(h[a:max(b,a+1)].mean(0))
    if s% (CH*20)==0: print(f"  affine {s}/{nsec} ({time.time()-t0:.0f}s)",flush=True)
audioAffect=np.array(emb,np.float32); print("audioAffect",audioAffect.shape,f"({time.time()-t0:.0f}s)",flush=True)
del m
# ---- audio low-level: per-second acoustic (RMS, centroid, bandwidth, rolloff, ZCR, 8 log-mel) ----
def melfb(nf,sr,nm=8,fmin=50,fmax=8000):
    def h2m(f): return 2595*np.log10(1+f/700); 
    def m2h(m): return 700*(10**(m/2595)-1)
    mp=np.linspace(h2m(fmin),h2m(fmax),nm+2); hz=m2h(mp); bins=np.floor((nf+1)*hz/sr).astype(int)
    fb=np.zeros((nm,nf//2+1))
    for i in range(1,nm+1):
        l,c,r=bins[i-1],bins[i],bins[i+1]
        for j in range(l,c): fb[i-1,j]=(j-l)/max(c-l,1)
        for j in range(c,r): fb[i-1,j]=(r-j)/max(r-c,1)
    return fb
FR=400; HOP=160; win=np.hanning(FR); fb=melfb(FR,SR)
low=[]
for s in range(nsec):
    seg=aud[s*SR:(s+1)*SR]
    if len(seg)<FR: low.append(np.zeros(13)); continue
    frames=[seg[i:i+FR]*win for i in range(0,len(seg)-FR,HOP)]
    F=np.abs(np.fft.rfft(np.array(frames),axis=1))+1e-8; P=F**2
    freqs=np.fft.rfftfreq(FR,1/SR)
    rms=np.sqrt((np.array(frames)**2).mean(1)).mean()
    cent=((P*freqs).sum(1)/P.sum(1)).mean()
    bw=(np.sqrt((P*(freqs[None,:]-((P*freqs).sum(1)/P.sum(1))[:,None])**2).sum(1)/P.sum(1))).mean()
    cs=np.cumsum(P,1); roll=(freqs[(cs>=0.85*cs[:,-1:]).argmax(1)]).mean()
    zcr=(np.abs(np.diff(np.sign(np.array(frames)),axis=1))>0).mean()
    logmel=np.log((P@fb.T)+1e-6).mean(0)
    low.append(np.concatenate([[rms,cent,bw,roll,zcr],logmel]))
audioLow=np.array(low,np.float32); print("audioLow",audioLow.shape,f"({time.time()-t0:.0f}s)",flush=True)
np.savez(f"{A}/audio_features.npz", audioAffect=audioAffect, audioLow=audioLow)
print("SAVED audio_features.npz",f"({time.time()-t0:.0f}s)",flush=True)
