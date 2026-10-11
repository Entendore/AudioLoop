#!/usr/bin/env python3
"""
Audio Looper v3 — PySide6 application for creating and playing audio loops.

v3 additions: Minimap overview bar, named markers, right-click context menu,
  keyboard nudge for loop points, jump forward/backward, audio normalization,
  silence-based loop suggestions, auto-follow playhead, vertical zoom,
  Help > Keyboard Shortcuts dialog, copy times to clipboard.
"""

import sys, os, json, math, time as _time
import numpy as np
import soundfile as sf

from PySide6.QtCore import (Qt, QUrl, QTimer, QRectF, QPointF, Signal, QSize, qWarning)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPainterPath, QLinearGradient,
    QDragEnterEvent, QDropEvent, QMouseEvent, QWheelEvent, QShortcut,
    QAction, QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QGroupBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QToolBar, QStatusBar, QSplitter, QFrame,
    QSizePolicy, QMessageBox, QListWidget, QListWidgetItem, QToolTip,
    QMenu, QMenuBar, QDialog, QTextBrowser, QDialogButtonBox,
    QInputDialog,
)
try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
except ImportError:
    QMediaPlayer = None
    QAudioOutput = None

SETTINGS_FILE = os.path.expanduser("~/.audio_looper_settings.json")
MAX_RECENT = 10
MAX_UNDO = 50
MAX_MARKERS = 100


def detect_bpm(data: np.ndarray, sr: int) -> float:
    if data.ndim == 2: data = np.mean(data, axis=1)
    data = data.astype(np.float64)
    fs, hop = 1024, 512
    nf = (len(data) - fs) // hop
    if nf < 10: return 0.0
    w = np.hanning(fs); prev = None; onset = np.zeros(nf)
    for i in range(nf):
        fr = data[i*hop:i*hop+fs] * w; sp = np.abs(np.fft.rfft(fr))
        if prev is not None: onset[i] = np.sum(np.maximum(sp - prev, 0))
        prev = sp
    onset = onset[1:]
    if onset.max() <= 0: return 0.0
    onset /= onset.max()
    mn = max(1, int(60/240*sr/hop)); mx = min(len(onset)-1, int(60/30*sr/hop))
    if mn >= mx: return 0.0
    c = np.zeros(mx - mn)
    for l in range(mn, mx): c[l-mn] = np.sum(onset[l:] * onset[:len(onset)-l])
    bi = int(np.argmax(c)); bl = bi + mn
    bs = bl * hop / sr
    if bs <= 0: return 0.0
    bpm = 60.0 / bs
    if bpm < 70:
        dl = bl // 2
        if dl >= mn and dl-mn < len(c) and c[dl-mn] > c[bi]*0.5: bpm *= 2; bl = dl
    elif bpm > 180:
        hl = bl * 2
        if hl < mx and hl-mn < len(c) and c[hl-mn] > c[bi]*0.5: bpm /= 2; bl = hl
    return round(bpm, 1) if 40 <= bpm <= 300 else 0.0


def detect_silence_regions(data, sr, threshold_db=-40, min_dur=0.3):
    if data.ndim == 2: data = np.mean(data, axis=1)
    fsz = int(sr*0.02); h = fsz//2; n = len(data)
    if n < fsz*2: return []
    rms = np.array([np.sqrt(np.mean(data[i:i+fsz]**2)) for i in range(0, n-fsz, h)])
    th = 10**(threshold_db/20.0)
    sil = rms < th; regions = []; in_s = False; sf_ = 0
    for i, s in enumerate(sil):
        if s and not in_s: in_s = True; sf_ = i
        elif not s and in_s:
            in_s = False; ss = sf_*h/sr; es = i*h/sr
            if es - ss >= min_dur: regions.append((round(ss,3), round(es,3)))
    if in_s:
        ss = sf_*h/sr; es = len(sil)*h/sr
        if es - ss >= min_dur: regions.append((round(ss,3), round(es,3)))
    return regions


class LevelMeterWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rl = self._rr = 0.0; self._pl = self._pr = 0.0
        self._phl = self._phr = 0.0; self._hf = 0; self._hm = 30
        self.setMinimumWidth(24); self.setMaximumWidth(40)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setContentsMargins(0,0,0,0)
    def set_levels(self, rl, rr, pl, pr):
        self._rl=max(0,min(1,rl)); self._rr=max(0,min(1,rr)); self._pl=max(0,min(1,pl)); self._pr=max(0,min(1,pr))
        if self._pl>=self._phl: self._phl=self._pl; self._hf=0
        else: self._hf+=1
        if self._hf>self._hm: self._phl=max(self._pl,self._phl-0.02)
        if self._pr>=self._phr: self._phr=self._pr
        else:
            if self._hf>self._hm: self._phr=max(self._pr,self._phr-0.02)
        self.update()
    def reset(self):
        self._rl=self._rr=self._pl=self._pr=self._phl=self._phr=0.0; self.update()
    def paintEvent(self, e):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing,False)
        w,h=self.width(),self.height(); p.fillRect(self.rect(),QColor(12,12,18))
        bw=max((w-6)//2,4)
        self._bar(p,2,2,bw,h-4,self._rl,self._phl); self._bar(p,2+bw+4,2,bw,h-4,self._rr,self._phr)
        p.setPen(QColor(120,120,140)); p.setFont(QFont("Monospace",7))
        p.drawText(QRectF(0,0,bw+2,12),Qt.AlignmentFlag.AlignCenter,"L")
        p.drawText(QRectF(bw+4,0,bw+2,12),Qt.AlignmentFlag.AlignCenter,"R"); p.end()
    def _bar(self,p,x,y,w,h,rms,pk):
        p.fillRect(x,y,w,h,QColor(20,20,30)); rh=int(h*rms)
        for i in range(rh):
            r=i/h; c=QColor(40,180,80) if r<0.6 else QColor(220,200,40) if r<0.85 else QColor(220,50,40)
            p.setPen(c); p.drawLine(x,y+h-1-i,x+w,y+h-1-i)
        py=y+h-1-int(h*pk)
        if 0<=py-y<=h: p.setPen(QColor(255,255,255)); p.drawLine(x,py,x+w,py)
        p.setPen(QColor(60,60,80)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawRect(x,y,w,h)


class MinimapWidget(QWidget):
    viewportClicked = Signal(float, float)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data=None; self._sr=44100; self._dur=0.0
        self._vs=0.0; self._ve=1.0; self._ls=-1.0; self._le=-1.0; self._pp=0.0
        self._cache=None; self._drag=False; self._dx=0; self._dvs=0.0; self._dve=0.0
        self.setFixedHeight(48); self.setMinimumHeight(36); self.setMouseTracking(True)
    def set_audio(self, data, sr):
        self._data=data; self._sr=sr; self._dur=len(data)/sr; self._cache=None; self.update()
    def set_viewport(self, s, e): self._vs=s; self._ve=e; self.update()
    def set_loop_region(self, s, e): self._ls=s; self._le=e; self.update()
    def set_playback_position(self, p): self._pp=p; self.update()
    def _build(self, pw):
        if self._data is None or pw<=0: return np.zeros(pw)
        d=self._data
        if d.ndim==2: d=np.mean(d,axis=1)
        n=len(d); idx=np.linspace(0,n,pw,dtype=int); t=np.zeros(pw)
        for i in range(pw-1):
            c=d[idx[i]:idx[i+1]]
            if len(c)>0: t[i]=np.max(np.abs(c))
        if pw>0:
            c=d[idx[-1]:]
            if len(c)>0: t[-1]=np.max(np.abs(c))
        mx=t.max()
        if mx>0: t/=mx
        self._cache=t; return t
    def paintEvent(self, e):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing,False)
        w,h=self.width(),self.height(); p.fillRect(self.rect(),QColor(12,12,20))
        if self._data is None or self._dur<=0:
            p.setPen(QColor(60,60,80)); p.setFont(QFont("Sans",8))
            p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,"No audio"); p.end(); return
        ml,mr=4,4; pw=w-ml-mr
        if pw<=0: p.end(); return
        if self._cache is None or len(self._cache)!=pw: self._build(pw)
        env=self._cache; mid=h/2
        g=QLinearGradient(0,0,0,h); g.setColorAt(0,QColor(0,120,200,120)); g.setColorAt(0.5,QColor(0,120,200,40)); g.setColorAt(1,QColor(0,120,200,120))
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(g))
        for i in range(pw):
            x=ml+i; bh=int(env[i]*(h/2)*0.85); p.drawRect(x,int(mid-bh),1,bh*2)
        if self._ls>=0 and self._le>self._ls:
            x1=ml+int(self._ls/self._dur*pw); x2=ml+int(self._le/self._dur*pw)
            p.fillRect(x1,0,x2-x1,h,QColor(255,165,0,40))
            p.setPen(QPen(QColor(255,165,0,150),1)); p.drawLine(x1,0,x1,h); p.drawLine(x2,0,x2,h)
        vx1=ml+int(self._vs/self._dur*pw); vx2=ml+int(self._ve/self._dur*pw)
        p.fillRect(vx1,0,vx2-vx1,h,QColor(255,255,255,15))
        p.setPen(QPen(QColor(200,200,220),1.5)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawRect(vx1,0,vx2-vx1,h-1)
        if 0<=self._pp<=self._dur:
            px=ml+int(self._pp/self._dur*pw); p.setPen(QPen(QColor(255,60,60),1)); p.drawLine(px,0,px,h)
        p.end()
    def _x2s(self, x):
        w=self.width()-8
        return max(0,min(self._dur,(x-4)/w*self._dur)) if w>0 else 0
    def mousePressEvent(self, e):
        if e.button()==Qt.MouseButton.LeftButton and self._dur>0:
            self._drag=True; self._dx=int(e.position().x()); self._dvs=self._vs; self._dve=self._ve
            s=self._x2s(self._dx); sp=self._ve-self._vs
            self._vs=max(0,s-sp/2); self._ve=min(self._dur,s+sp/2)
            if self._ve-self._vs<sp:
                if self._vs==0: self._ve=self._vs+sp
                else: self._vs=self._ve-sp
            self.viewportClicked.emit(self._vs,self._ve)
    def mouseMoveEvent(self, e):
        if self._drag and self._dur>0:
            x=int(e.position().x()); dx=x-self._dx; w=self.width()-8
            if w<=0: return
            ds=dx/w*self._dur; sp=self._dve-self._dvs; ns=self._dvs+ds; ne=ns+sp
            if ns<0: ne-=ns; ns=0
            if ne>self._dur: ns-=(ne-self._dur); ne=self._dur
            self._vs=max(0,ns); self._ve=ne; self.viewportClicked.emit(ns,ne); self.update()
    def mouseReleaseEvent(self, e): self._drag=False


class LoopPresetsPanel(QWidget):
    presetSelected = Signal(float, float); clearRequested = Signal()
    def __init__(self, parent=None):
        super().__init__(parent); self._p=[]
        la=QVBoxLayout(self); la.setContentsMargins(4,4,4,4); la.setSpacing(4)
        h=QLabel("Loop Presets"); h.setFont(QFont("Sans",10,QFont.Weight.Bold)); h.setAlignment(Qt.AlignmentFlag.AlignCenter)
        la.addWidget(h)
        self._l=QListWidget(); self._l.setFixedWidth(170)
        self._l.setStyleSheet("QListWidget{background:#161620;border:1px solid #2a2a40;border-radius:4px;color:#c0c0d0;font-size:11px;}"
            "QListWidget::item{padding:4px;border-bottom:1px solid #1e1e30;}"
            "QListWidget::item:selected{background:#2a2a50;}QListWidget::item:hover{background:#222240;}")
        la.addWidget(self._l)
        r=QHBoxLayout(); ba=QPushButton("Save"); br=QPushButton("Remove"); ba.setFixedHeight(28); br.setFixedHeight(28)
        r.addWidget(ba); r.addWidget(br); la.addLayout(r)
        bc=QPushButton("Clear All"); bc.setFixedHeight(28); la.addWidget(bc)
        ba.clicked.connect(self._add); br.clicked.connect(self._rem); bc.clicked.connect(self._clr)
        self._l.itemDoubleClicked.connect(self._dbl)
        self.setFixedWidth(185)
    def save_current(self, s, e): self._ps=s; self._pe=e
    def _add(self):
        if not hasattr(self,'_ps') or self._ps<0: return
        d=self._pe-self._ps; n=f"{d:.1f}s loop" if d>=1 else f"Loop {len(self._p)+1}"
        self._p.append({"name":n,"start":round(self._ps,3),"end":round(self._pe,3)}); self._rf(); self._l.setCurrentRow(len(self._p)-1)
    def _rem(self):
        r=self._l.currentRow()
        if 0<=r<len(self._p): self._p.pop(r); self._rf()
    def _clr(self): self._p.clear(); self._rf(); self.clearRequested.emit()
    def _dbl(self, i):
        r=self._l.row(i)
        if 0<=r<len(self._p): pr=self._p[r]; self.presetSelected.emit(pr["start"],pr["end"])
    def _rf(self):
        self._l.clear()
        for pr in self._p: self._l.addItem(f"{pr['name']}  ({pr['end']-pr['start']:.2f}s)")
    def update_save_pending(self, s, e): self._ps=s; self._pe=e


class MarkersPanel(QWidget):
    markerJumpRequested = Signal(float); markerSetRequested = Signal()
    def __init__(self, parent=None):
        super().__init__(parent); self._m=[]
        la=QVBoxLayout(self); la.setContentsMargins(4,4,4,4); la.setSpacing(4)
        h=QLabel("Markers"); h.setFont(QFont("Sans",10,QFont.Weight.Bold)); h.setAlignment(Qt.AlignmentFlag.AlignCenter)
        la.addWidget(h)
        self._l=QListWidget(); self._l.setFixedWidth(170)
        self._l.setStyleSheet("QListWidget{background:#161620;border:1px solid #2a2a40;border-radius:4px;color:#c0c0d0;font-size:11px;}"
            "QListWidget::item{padding:4px;border-bottom:1px solid #1e1e30;}"
            "QListWidget::item:selected{background:#2a2a50;}QListWidget::item:hover{background:#222240;}")
        la.addWidget(self._l)
        r=QHBoxLayout(); ba=QPushButton("Add"); bah=QPushButton("Add Here"); bd=QPushButton("Del")
        ba.setFixedHeight(28); bah.setFixedHeight(28); bd.setFixedHeight(28)
        r.addWidget(ba); r.addWidget(bah); r.addWidget(bd); la.addLayout(r)
        bj=QPushButton("Jump To"); bj.setFixedHeight(28); la.addWidget(bj)
        ba.clicked.connect(self._add_dlg); bah.clicked.connect(self.markerSetRequested); bd.clicked.connect(self._rem)
        bj.clicked.connect(lambda: self._jump(self._l.currentRow()))
        self._l.itemDoubleClicked.connect(lambda i: self._jump(self._l.row(i)))
        self.setFixedWidth(185)
    def add_marker(self, name, t):
        if len(self._m)>=MAX_MARKERS: return
        self._m.append({"name":name,"time":round(t,3)}); self._m.sort(key=lambda m:m["time"]); self._rf()
    def get_markers(self): return list(self._m)
    def clear_markers(self): self._m.clear(); self._rf()
    def _add_dlg(self):
        n,ok=QInputDialog.getText(self,"Add Marker","Name:",text=f"M{len(self._m)+1}")
        if ok and n.strip():
            t,ok2=QInputDialog.getDouble(self,"Time","Seconds:",0,0,99999,3)
            if ok2: self.add_marker(n.strip(),t)
    def _rem(self):
        r=self._l.currentRow()
        if 0<=r<len(self._m): self._m.pop(r); self._rf()
    def _jump(self, r):
        if 0<=r<len(self._m): self.markerJumpRequested.emit(self._m[r]["time"])
    def _rf(self):
        self._l.clear()
        for m in self._m: self._l.addItem(f"{m['name']}  @ {m['time']:.2f}s")


class WaveformWidget(QWidget):
    loopRegionChanged = Signal(float, float)
    positionClicked = Signal(float)
    hoverTimeChanged = Signal(float)
    nudgeRequested = Signal(str, float)
    contextMenuRequested = Signal(float)
    markerAddRequested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data=None; self._sr=44100; self._dur=0.0
        self._vs=0.0; self._ve=1.0; self._ls=-1.0; self._le=-1.0; self._pp=0.0
        self._spec_cache=None; self._spec_dirty=True; self._show_spec=False
        self._bpm=0.0; self._show_bg=False; self._snap=False
        self._drag=None; self._dx=0; self._ds=0.0; self._dls=0.0; self._dle=0.0; self._hx=-1
        self._markers=[]; self._vzoom=1.0
        self._wc=QColor(0,180,255); self._wbg=QColor(18,18,24)
        self._lrc=QColor(255,165,0,50); self._lbc=QColor(255,165,0)
        self._phc=QColor(255,60,60); self._cc=QColor(60,60,80)
        self._ruc=QColor(100,100,120); self._tc=QColor(200,200,210)
        self._bgc=QColor(100,80,200,60); self._mc=QColor(0,230,120)
        self._dt=None; self._db=None; self._dvs=-1; self._dve=-1
        self.setMinimumHeight(120); self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True); self.setAcceptDrops(True); self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_audio(self, d, sr):
        self._data=d; self._sr=sr; self._dur=len(d)/sr; self._vs=0; self._ve=self._dur
        self._ls=-1; self._le=-1; self._pp=0; self._spec_cache=None; self._spec_dirty=True
        self._dt=None; self._db=None; self._dvs=-1; self._dve=-1; self.update()
    def set_playback_position(self, p): self._pp=p; self.update()
    def set_loop_region(self, s, e): self._ls=s; self._le=e; self.update(); self.loopRegionChanged.emit(s,e)
    def clear_loop_region(self): self._ls=-1; self._le=-1; self.update(); self.loopRegionChanged.emit(-1,-1)
    def get_loop_region(self):
        return (self._ls,self._le) if self._ls>=0 and self._le>self._ls else (-1,-1)
    def set_show_spectrogram(self, v): self._show_spec=v; self.update()
    def set_bpm(self, b): self._bpm=b
    def set_show_beat_grid(self, v): self._show_bg=v; self.update()
    def set_snap_to_beat(self, v): self._snap=v
    def set_markers(self, m): self._markers=m; self.update()
    def snap_to_nearest_beat(self, sec):
        if not self._snap or self._bpm<=0: return sec
        bd=60.0/self._bpm
        if bd<=0: return sec
        return max(0,min(self._dur,round(sec/bd)*bd))
    def zoom_in(self):
        c=(self._vs+self._ve)/2; h=(self._ve-self._vs)/4; self._vs=max(0,c-h); self._ve=min(self._dur,c+h)
        self._spec_dirty=True; self.update()
    def zoom_out(self):
        c=(self._vs+self._ve)/2; h=self._ve-self._vs; self._vs=max(0,c-h); self._ve=min(self._dur,c+h)
        self._spec_dirty=True; self.update()
    def zoom_fit(self):
        self._vs=0; self._ve=self._dur if self._dur>0 else 1; self._spec_dirty=True; self.update()
    def zoom_to_region(self, s, e):
        if s<e: self._vs=max(0,s); self._ve=min(self._dur,e); self._spec_dirty=True; self.update()
    def scroll_left(self, f=0.1):
        s=self._ve-self._vs; self._vs=max(0,self._vs-s*f); self._ve=self._vs+s; self._spec_dirty=True; self.update()
    def scroll_right(self, f=0.1):
        s=self._ve-self._vs; self._ve=min(self._dur,self._ve+s*f); self._vs=self._ve-s; self._spec_dirty=True; self.update()
    def _x2s(self, x):
        w=self.width(); pw=w-20
        return self._vs+(x-10)/pw*(self._ve-self._vs) if pw>0 else self._vs
    def _s2x(self, s):
        w=self.width(); pw=w-20
        return int(10+(s-self._vs)/(self._ve-self._vs)*pw) if pw>0 and (self._ve-self._vs)>0 else 10
    def _hit(self, x):
        ht=8; ls=self._s2x(self._ls) if self._ls>=0 else -9999; le=self._s2x(self._le) if self._le>=0 else -9999
        if abs(x-ls)<=ht: return "ls"
        if abs(x-le)<=ht: return "le"
        return None
    def _inloop(self, x):
        return self._ls>=0 and self._s2x(self._ls)<=x<=self._s2x(self._le)
    def _cds(self, pw):
        sr=self._sr; s=max(0,int(self._vs*sr)); e=min(len(self._data),int(self._ve*sr))
        if e<=s or pw<=0: return np.zeros(pw),np.zeros(pw)
        d=self._data[s:e]
        if d.ndim==2: d=np.mean(d,axis=1)
        n=len(d); idx=np.linspace(0,n,pw+1,dtype=int)
        t=np.full(pw,-1e30); b=np.full(pw,1e30)
        for i in range(pw):
            c=d[idx[i]:idx[i+1]]
            if len(c)>0: t[i]=np.max(c); b[i]=np.min(c)
        return t, b
    def _cspec(self, pw, ph):
        if self._data is None: return None
        if not self._spec_dirty and self._spec_cache is not None: return self._spec_cache
        sr=self._sr; s=max(0,int(self._vs*sr)); e=min(len(self._data),int(self._ve*sr))
        d=self._data[s:e]
        if d.ndim==2: d=np.mean(d,axis=1)
        n=len(d)
        if n<256: self._spec_dirty=False; return None
        fs=1024; hop=max(1,n//pw); nf=fs//2+1
        nfr=min(pw,(n-fs)//hop+1) if n>=fs else 0
        if nfr<=0: self._spec_dirty=False; return None
        w=np.hanning(fs); sp=np.zeros((nf,nfr),dtype=np.float32)
        for i in range(nfr):
            st=i*hop; fr=d[st:st+fs]
            if len(fr)<fs: fr=np.pad(fr,(0,fs-len(fr)))
            sp[:,i]=np.abs(np.fft.rfft(fr*w))
        sp=20*np.log10(sp+1e-10); mn,mx=sp.min(),sp.max()
        sp=(sp-mn)/(mx-mn) if mx>mn else np.zeros_like(sp)
        th=min(ph,nf); st=max(1,nf//th); sp=sp[::st,:]
        self._spec_cache=sp; self._spec_dirty=False; return sp

    def paintEvent(self, e):
        if self._data is None: self._empty(); return
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing,True)
        self._pbg(p); self._pruler(p)
        if self._show_bg and self._bpm>0: self._pbeats(p)
        p.setPen(QPen(self._cc,1,Qt.PenStyle.DashLine)); cy=(24+self.height())/2; p.drawLine(0,int(cy),self.width(),int(cy))
        if self._show_spec: self._pspec(p)
        else: self._pwave(p)
        self._pmark(p); self._ploop(p); self._pph(p); self._pinfo(p); self._ptip(p)
        p.end()
    def _empty(self):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing,True)
        p.fillRect(self.rect(),self._wbg); p.setPen(self._tc); f=QFont("Sans",14); f.setItalic(True); p.setFont(f)
        p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,"Drop audio here\nor click Load Audio"); p.end()
    def _pbg(self, p): p.fillRect(self.rect(),self._wbg)
    def _pruler(self, p):
        rh=24; p.fillRect(0,0,self.width(),rh,QColor(24,24,32))
        span=self._ve-self._vs; ri=span/15; ivs=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,30,60]
        iv=ivs[-1]
        for v in ivs:
            if v>=ri: iv=v; break
        p.setPen(QPen(self._ruc,1)); p.setFont(QFont("Monospace",8)); p.setClipRect(0,0,self.width(),rh)
        t=(int(self._vs/iv))*iv
        while t<=self._ve:
            x=self._s2x(t)
            if 0<=x<=self.width(): p.drawLine(x,rh-6,x,rh); p.drawText(x+2,rh-8,f"{t:.2f}s")
            t+=iv
        p.setClipping(False)
    def _pbeats(self, p):
        rh=24; bd=60.0/self._bpm
        if bd<=0: return
        fb=math.ceil(self._vs/bd); lb=math.floor(self._ve/bd)
        pen=QPen(self._bgc,1,Qt.PenStyle.DotLine); p.setPen(pen)
        for b in range(int(fb),int(lb)+1):
            x=self._s2x(b*bd)
            if 0<=x<=self.width():
                p.drawLine(x,rh,x,self.height())
                p.setPen(QColor(120,100,220,140)); p.setFont(QFont("Monospace",7)); p.drawText(x+2,rh+12,str(b+1)); p.setPen(pen)
    def _pwave(self, p):
        if self._data is None: return
        rh=24; pw=self.width()-20
        if pw<=0: return
        dt=rh+4; db=self.height()-4; dh=db-dt; cy=(dt+db)/2
        if self._dt is None or abs(self._dvs-self._vs)>1e-6 or abs(self._dve-self._ve)>1e-6:
            self._dt,self._db=self._cds(int(pw)); self._dvs=self._vs; self._dve=self._ve
        tops=self._dt*self._vzoom; bots=self._db*self._vzoom
        apw=len(tops); xs=np.arange(apw)+10
        yt=cy-tops*(dh/2)*0.9; yb=cy-bots*(dh/2)*0.9
        pt=QPainterPath(); pb=QPainterPath()
        for i in range(apw):
            x,a,b=int(xs[i]),float(yt[i]),float(yb[i])
            if i==0: pt.moveTo(x,a); pb.moveTo(x,b)
            else: pt.lineTo(x,a); pb.lineTo(x,b)
        fp=QPainterPath(pt); fp.lineTo(int(xs[-1]),cy); fp.connectPath(pb.toReversed()); fp.closeSubpath()
        g=QLinearGradient(0,dt,0,db); wc=self._wc
        g.setColorAt(0,QColor(wc.red(),wc.green(),wc.blue(),180)); g.setColorAt(0.5,QColor(wc.red(),wc.green(),wc.blue(),60))
        g.setColorAt(1,QColor(wc.red(),wc.green(),wc.blue(),180))
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(g)); p.drawPath(fp)
        p.setPen(QPen(wc,1)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawPath(pt); p.drawPath(pb)
    def _pspec(self, p):
        rh=24; pw=self.width()-20; ph=self.height()-rh-8
        if pw<=0 or ph<=0: return
        sp=self._cspec(int(pw),ph)
        if sp is None: return
        nf,nfr=sp.shape; cw=max(1,pw/nfr); ch=max(1,ph/nf)
        for j in range(nfr):
            x=10+int(j*cw)
            for i in range(nf):
                y=rh+4+int(i*ch); v=sp[i,j]
                if v<0.25: r,g,b=0,0,int(v*4*180)
                elif v<0.5: t=(v-0.25)*4; r,g,b=0,int(t*200),180
                elif v<0.75: t=(v-0.5)*4; r,g,b=int(t*255),200,int(180-t*180)
                else: t=(v-0.75)*4; r,g,b=255,int(200-t*200),0
                p.fillRect(x,y,max(1,int(cw)+1),max(1,int(ch)+1),QColor(r,g,b))
    def _pmark(self, p):
        if not self._markers: return
        rh=24
        for m in self._markers:
            x=self._s2x(m["time"])
            if 0<=x<=self.width():
                p.setPen(QPen(self._mc,1,Qt.PenStyle.DashLine)); p.drawLine(x,rh,x,self.height())
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(self._mc))
                tri=QPainterPath(); tri.moveTo(x,rh+2); tri.lineTo(x+8,rh+2); tri.lineTo(x,rh+11); tri.closeSubpath()
                p.drawPath(tri)
                p.setPen(self._mc); p.setFont(QFont("Sans",7)); p.drawText(x+10,rh+11,m["name"])
    def _ploop(self, p):
        if self._ls<0 or self._le<=self._ls: return
        rh=24; lx=self._s2x(self._ls); rx=self._s2x(self._le)
        p.fillRect(QRectF(0,rh,lx,self.height()-rh),QColor(0,0,0,80))
        p.fillRect(QRectF(rx,rh,self.width()-rx,self.height()-rh),QColor(0,0,0,80))
        rect=QRectF(lx,rh,rx-lx,self.height()-rh); p.fillRect(rect,self._lrc)
        p.setPen(QPen(self._lbc,2)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawRect(rect)
        for hx in [lx,rx]:
            p.setBrush(QBrush(self._lbc)); p.setPen(Qt.PenStyle.NoPen); cy=(rh+self.height())/2
            p.drawRect(QRectF(hx-3,cy-15,6,30))
            for (my,dy) in [(rh,10),(self.height(),-10)]:
                t=QPainterPath(); t.moveTo(hx,my); t.lineTo(hx-6,my+dy); t.lineTo(hx+6,my+dy); t.closeSubpath(); p.drawPath(t)
        p.setPen(self._lbc); p.setFont(QFont("Monospace",8,QFont.Weight.Bold))
        p.drawText(QPointF(lx+3,rh+14),f"A {self._ls:.2f}s"); p.drawText(QPointF(rx-70,rh+14),f"B {self._le:.2f}s")
        dur=self._le-self._ls; cx=(lx+rx)/2
        p.setPen(QColor(255,200,100)); p.setFont(QFont("Sans",10,QFont.Weight.Bold))
        p.drawText(QPointF(cx-20,rh+28),f"{dur:.2f}s")
        if self._bpm>0:
            beats=dur/(60.0/self._bpm); p.setPen(QColor(200,180,255)); p.setFont(QFont("Sans",9))
            p.drawText(QPointF(cx-20,rh+42),f"{beats:.1f} beats")
    def _pph(self, p):
        if self._dur<=0: return
        rh=24; x=self._s2x(self._pp)
        if x<0 or x>self.width(): return
        for w,a in [(4,30),(2,80)]: p.setPen(QPen(QColor(255,60,60,a),w)); p.drawLine(x,rh,x,self.height())
        p.setPen(QPen(self._phc,1.5)); p.drawLine(x,rh,x,self.height())
        tri=QPainterPath(); tri.moveTo(x-6,rh); tri.lineTo(x+6,rh); tri.lineTo(x,rh+9); tri.closeSubpath()
        p.setBrush(QBrush(self._phc)); p.setPen(Qt.PenStyle.NoPen); p.drawPath(tri)
    def _pinfo(self, p):
        if self._dur<=0: return
        p.setPen(self._tc); p.setFont(QFont("Monospace",9))
        bs=f"  |  BPM: {self._bpm:.1f}" if self._bpm>0 else ""
        vz=f"  |  VZoom: {self._vzoom:.1f}x" if self._vzoom!=1.0 else ""
        p.drawText(12,self.height()-8,f"Duration: {self._dur:.2f}s  |  SR: {self._sr}Hz  |  View: {self._vs:.2f}s - {self._ve:.2f}s{bs}{vz}")
    def _ptip(self, p):
        if self._hx<0 or self._data is None: return
        sec=self._x2s(self._hx)
        if sec<0 or sec>self._dur: return
        m=int(sec//60); s=sec%60; txt=f"{m}:{s:05.2f}"
        if self._bpm>0 and self._snap: txt+=f"  (beat {sec/(60.0/self._bpm):.1f})"
        p.setFont(QFont("Monospace",9)); tw=p.fontMetrics().horizontalAdvance(txt)+12
        tx=min(self._hx+15,self.width()-tw-5); ty=30
        p.setPen(QColor(80,80,100)); p.setBrush(QBrush(QColor(30,30,45,220)))
        p.drawRoundedRect(QRectF(tx,ty,tw,20),4,4); p.setPen(QColor(220,220,240)); p.drawText(QPointF(tx+6,ty+14),txt)

    def mousePressEvent(self, e):
        if e.button()==Qt.MouseButton.LeftButton:
            x=int(e.position().x()); h=self._hit(x)
            if h: self._drag=h; self._dx=x
            elif self._inloop(x):
                self._drag="region"; self._dx=x; self._ds=self._x2s(x); self._dls=self._ls; self._dle=self._le
            else:
                sec=self.snap_to_nearest_beat(self._x2s(x)); self.positionClicked.emit(max(0,min(self._dur,sec)))
        elif e.button()==Qt.MouseButton.RightButton:
            self.contextMenuRequested.emit(self._x2s(int(e.position().x())))
    def mouseMoveEvent(self, e):
        x=int(e.position().x()); self._hx=x
        if self._data is not None: self.hoverTimeChanged.emit(self._x2s(x))
        h=self._hit(x)
        if h: self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif self._inloop(x): self.setCursor(Qt.CursorShape.OpenHandCursor)
        else: self.setCursor(Qt.CursorShape.CrossCursor)
        if self._drag is None: self.update(); return
        sec=self.snap_to_nearest_beat(self._x2s(x)); sec=max(0,min(self._dur,sec))
        if self._drag=="ls":
            self._ls=min(sec,self._le-0.01); self.update(); self.loopRegionChanged.emit(self._ls,self._le)
        elif self._drag=="le":
            self._le=max(sec,self._ls+0.01); self.update(); self.loopRegionChanged.emit(self._ls,self._le)
        elif self._drag=="region":
            dx=sec-self._ds; ns=self._dls+dx; ne=self._dle+dx
            if ns<0: ne-=ns; ns=0
            if ne>self._dur: ns-=(ne-self._dur); ne=self._dur
            if ns>=0 and ne>ns: self._ls=ns; self._le=ne; self.update(); self.loopRegionChanged.emit(ns,ne)
    def mouseReleaseEvent(self, e):
        if self._drag: self._drag=None; self.setCursor(Qt.CursorShape.ArrowCursor)
    def mouseDoubleClickEvent(self, e):
        if e.button()==Qt.MouseButton.LeftButton:
            x=int(e.position().x()); sec=self.snap_to_nearest_beat(self._x2s(x))
            self.positionClicked.emit(max(0,min(self._dur,sec)))
        elif e.button()==Qt.MouseButton.RightButton:
            self.markerAddRequested.emit(self._x2s(int(e.position().x())))
    def leaveEvent(self, e): self._hx=-1; self.update()
    def wheelEvent(self, e):
        x=int(e.position().x()); sec=self._x2s(x); mod=e.modifiers()
        if mod & Qt.KeyboardModifier.ControlModifier:
            if e.angleDelta().y()>0: self._vzoom=min(5.0,self._vzoom*1.2)
            else: self._vzoom=max(0.2,self._vzoom/1.2)
            self._dt=None; self.update()
        else:
            f=0.75 if e.angleDelta().y()>0 else 1.35
            ns=(self._ve-self._vs)*f; ns=max(0.01,min(self._dur,ns))
            r=(sec-self._vs)/(self._ve-self._vs) if (self._ve-self._vs)>0 else 0.5
            self._vs=sec-r*ns; self._ve=sec+(1-r)*ns
            if self._vs<0: self._ve-=self._vs; self._vs=0
            if self._ve>self._dur: self._vs-=(self._ve-self._dur); self._ve=self._dur
            self._vs=max(0,self._vs); self._spec_dirty=True
        self.update()
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        urls=e.mimeData().urls()
        if urls:
            pa=urls[0].toLocalFile()
            if os.path.isfile(pa):
                w=self.parent()
                while w and not isinstance(w,AudioLooperApp): w=w.parent()
                if w: w.load_file(pa)
    def keyPressEvent(self, e):
        k=e.key(); mod=e.modifiers()
        if mod & Qt.KeyboardModifier.ShiftModifier:
            n=0.001
            if mod & Qt.KeyboardModifier.ControlModifier: n=0.1
            elif mod & Qt.KeyboardModifier.AltModifier: n=0.01
            if k==Qt.Key.Key_Left: self.nudgeRequested.emit("start",-n); return
            if k==Qt.Key.Key_Right: self.nudgeRequested.emit("start",n); return
        if k==Qt.Key.Key_Left: self.scroll_left(0.05)
        elif k==Qt.Key.Key_Right: self.scroll_right(0.05)
        elif k in(Qt.Key.Key_Plus,Qt.Key.Key_Equal): self.zoom_in()
        elif k==Qt.Key.Key_Minus: self.zoom_out()
        elif k==Qt.Key.Key_Home: self.zoom_fit()
        else: super().keyPressEvent(e)


class AudioLooperApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Looper v3"); self.setMinimumSize(960,560); self.resize(1250,720)
        self._ap=None; self._ad=None; self._sr=44100; self._looping=False; self._li=0
        self._rf=[]; self._us=[]; self._rs=[]; self._tt=[]; self._af=False
        self._player=QMediaPlayer(self); self._ao=QAudioOutput(self)
        self._player.setAudioOutput(self._ao); self._player.setLoops(1); self._ao.setVolume(0.8)
        self._pt=QTimer(self); self._pt.setInterval(30); self._pt.timeout.connect(lambda:None)
        self._lt=QTimer(self); self._lt.setInterval(50)
        self._setup(); self._connect(); self._style(); self._load_sett(); self._build_recent()

    def _setup(self):
        c=QWidget(); self.setCentralWidget(c); ml=QVBoxLayout(c); ml.setContentsMargins(8,8,8,8); ml.setSpacing(4)
        mb=self.menuBar()
        mb.setStyleSheet("QMenuBar{background:#16161e;color:#c0c0d0;border-bottom:1px solid #222230;padding:2px;}"
            "QMenuBar::item:selected{background:#2a2a50;}QMenu{background:#1a1a28;color:#c0c0d0;border:1px solid #333348;}"
            "QMenu::item:selected{background:#2a2a50;}")
        fm=mb.addMenu("File")
        a=fm.addAction("Open Audio..."); a.setShortcut(QKeySequence.StandardKey.Open); a.triggered.connect(self._on_load)
        fm.addSeparator(); self._rm=fm.addMenu("Recent Files"); fm.addSeparator()
        a=fm.addAction("Exit"); a.setShortcut(QKeySequence.StandardKey.Quit); a.triggered.connect(self.close)
        em=mb.addMenu("Edit")
        a=em.addAction("Undo Loop"); a.setShortcut(QKeySequence.StandardKey.Undo); a.triggered.connect(self._on_undo)
        a=em.addAction("Redo Loop"); a.setShortcut(QKeySequence.StandardKey.Redo); a.triggered.connect(self._on_redo)
        em.addSeparator(); a=em.addAction("Clear Loop"); a.triggered.connect(self._on_cl)
        em.addSeparator(); a=em.addAction("Normalize Loop Region"); a.triggered.connect(self._on_norm_loop)
        a=em.addAction("Normalize Full Audio"); a.triggered.connect(self._on_norm_all)
        vm=mb.addMenu("View")
        self._a_sp=vm.addAction("Spectrogram"); self._a_sp.setCheckable(True); self._a_sp.triggered.connect(lambda c:(self._wf.set_show_spectrogram(c),setattr(self._wf,'_spec_dirty',True)))
        self._a_bg=vm.addAction("Beat Grid"); self._a_bg.setCheckable(True); self._a_bg.triggered.connect(lambda c:(self._wf.set_show_beat_grid(c),self._sb.showMessage(f"Beat grid: {'ON' if c else 'OFF'}")))
        self._a_sn=vm.addAction("Snap to Beats"); self._a_sn.setCheckable(True); self._a_sn.triggered.connect(lambda c:(self._wf.set_snap_to_beat(c),self._sb.showMessage(f"Snap: {'ON' if c else 'OFF'}")))
        self._a_af=vm.addAction("Auto-Follow"); self._a_af.setCheckable(True); self._a_af.triggered.connect(lambda c:(setattr(self,'_af',c),self._sb.showMessage(f"Auto-follow: {'ON' if c else 'OFF'}")))
        vm.addSeparator(); a=vm.addAction("Auto-Suggest Loop (Silence)"); a.triggered.connect(self._on_suggest)
        hm=mb.addMenu("Help"); a=hm.addAction("Keyboard Shortcuts"); a.triggered.connect(self._show_keys)
        tb=QToolBar("T"); tb.setMovable(False); tb.setIconSize(QSize(20,20)); self.addToolBar(tb)
        def B(t,tp=""): b=QPushButton(t); b.setFixedHeight(30); b.setToolTip(tp); tb.addWidget(b); return b
        self._b_load=B("Load Audio"); tb.addSeparator()
        self._b_play=B("Play"); self._b_play.setEnabled(False)
        self._b_pause=B("Pause"); self._b_pause.setEnabled(False)
        self._b_stop=B("Stop"); self._b_stop.setEnabled(False)
        self._b_sb=B("<<","Skip -1s ([])"); self._b_sb.setEnabled(False)
        self._b_sf=B(">>","Skip +1s (])"); self._b_sf.setEnabled(False)
        tb.addSeparator()
        self._b_loop=B("Loop: OFF"); self._b_loop.setCheckable(True); self._b_loop.setEnabled(False)
        self._b_sa=B("Set A","Set A (A)"); self._b_sa.setEnabled(False)
        self._b_sb2=B("Set B","Set B (B)"); self._b_sb2.setEnabled(False)
        self._b_cl=B("Clear Loop"); self._b_cl.setEnabled(False)
        tb.addSeparator()
        self._b_zi=B("Zoom+","Zoom In"); self._b_zo=B("Zoom-","Zoom Out"); self._b_fit=B("Fit","Fit")
        tb.addSeparator()
        self._b_tap=B("Tap BPM","Tap (T)"); self._b_dbpm=B("Detect BPM")
        tb.addSeparator()
        self._b_undo=B("Undo"); self._b_redo=B("Redo"); tb.addSeparator()
        self._b_exp=B("Export Loop"); self._b_exp.setEnabled(False)
        cs=QSplitter(Qt.Orientation.Horizontal); ml.addWidget(cs,stretch=1)
        lw=QWidget(); ll=QVBoxLayout(lw); ll.setContentsMargins(0,0,0,0); ll.setSpacing(3)
        self._mm=MinimapWidget(); ll.addWidget(self._mm)
        self._wf=WaveformWidget(); ll.addWidget(self._wf,stretch=1)
        bf=QFrame(); bl=QHBoxLayout(bf); bl.setContentsMargins(4,2,4,2)
        self._ss=QSlider(Qt.Orientation.Horizontal); self._ss.setRange(0,1000); self._ss.setEnabled(False)
        bl.addWidget(QLabel("Seek:")); bl.addWidget(self._ss,stretch=1)
        self._lt2=QLabel("0:00.00 / 0:00.00"); self._lt2.setFixedWidth(160)
        self._lt2.setAlignment(Qt.AlignmentFlag.AlignCenter); self._lt2.setFont(QFont("Monospace",10)); bl.addWidget(self._lt2)
        self._lm_w=LevelMeterWidget(); self._lm_w.setFixedWidth(36); bl.addWidget(self._lm_w)
        bl.addWidget(QLabel("Vol:"))
        self._vs=QSlider(Qt.Orientation.Horizontal); self._vs.setRange(0,100); self._vs.setValue(80); self._vs.setFixedWidth(100); bl.addWidget(self._vs)
        bl.addWidget(QLabel("Speed:"))
        self._sp=QDoubleSpinBox(); self._sp.setRange(0.25,4.0); self._sp.setSingleStep(0.05); self._sp.setValue(1.0); self._sp.setFixedWidth(75); bl.addWidget(self._sp)
        ll.addWidget(bf)
        lf=QFrame(); lil=QHBoxLayout(lf); lil.setContentsMargins(4,2,4,2)
        self._ll=QLabel("No loop region -- drag on waveform, or use Set A/B"); self._ll.setFont(QFont("Sans",9))
        lil.addWidget(self._ll,stretch=1); lil.addWidget(QLabel("Crossfade(ms):"))
        self._cf=QSpinBox(); self._cf.setRange(0,5000); self._cf.setSingleStep(10); self._cf.setFixedWidth(80); lil.addWidget(self._cf)
        lil.addWidget(QLabel("Repeat:"))
        self._rp=QSpinBox(); self._rp.setRange(0,1000); self._rp.setSpecialValueText("Inf"); self._rp.setFixedWidth(70); lil.addWidget(self._rp)
        ll.addWidget(lf); cs.addWidget(lw)
        rs=QSplitter(Qt.Orientation.Vertical)
        self._pp=LoopPresetsPanel(); rs.addWidget(self._pp)
        self._mp=MarkersPanel(); rs.addWidget(self._mp)
        rs.setStretchFactor(0,1); rs.setStretchFactor(1,1); cs.addWidget(rs)
        cs.setStretchFactor(0,1); cs.setStretchFactor(1,0)
        self._sb=QStatusBar(); self.setStatusBar(self._sb); self._sb.showMessage("Ready -- load an audio file")


    def _connect(self):
        self._b_load.clicked.connect(self._on_load)
        self._b_play.clicked.connect(self._on_play); self._b_pause.clicked.connect(self._on_pause); self._b_stop.clicked.connect(self._on_stop)
        self._b_sb.clicked.connect(lambda:self._skip(-1)); self._b_sf.clicked.connect(lambda:self._skip(1))
        self._b_loop.clicked.connect(self._on_tloop)
        self._b_sa.clicked.connect(self._on_sa); self._b_sb2.clicked.connect(self._on_sb)
        self._b_cl.clicked.connect(self._on_cl)
        self._b_zi.clicked.connect(self._wf.zoom_in); self._b_zo.clicked.connect(self._wf.zoom_out); self._b_fit.clicked.connect(self._wf.zoom_fit)
        self._b_tap.clicked.connect(self._on_tap); self._b_dbpm.clicked.connect(self._on_dbpm)
        self._b_undo.clicked.connect(self._on_undo); self._b_redo.clicked.connect(self._on_redo)
        self._b_exp.clicked.connect(self._on_export)
        self._ss.sliderMoved.connect(self._on_seek_sl)
        self._vs.valueChanged.connect(lambda v:self._ao.setVolume(v/100.0))
        self._sp.valueChanged.connect(lambda v:self._player.setPlaybackRate(v))
        self._wf.loopRegionChanged.connect(self._on_lc)
        self._wf.positionClicked.connect(self._on_sp)
        self._wf.hoverTimeChanged.connect(self._on_hov)
        self._wf.nudgeRequested.connect(self._on_nudge)
        self._wf.contextMenuRequested.connect(self._on_ctx)
        self._wf.markerAddRequested.connect(self._on_am)
        self._mm.viewportClicked.connect(self._on_mm_nav)
        self._player.positionChanged.connect(self._on_pc)
        self._player.playbackStateChanged.connect(self._on_sc)
        self._player.errorOccurred.connect(lambda e,s:(self._sb.showMessage(f"Error: {s}"),qWarning(f"Error: {s}")))
        self._lt.timeout.connect(self._on_lt_tick)
        self._pp.presetSelected.connect(self._on_ps)
        self._pp.clearRequested.connect(self._on_cl)
        self._mp.markerJumpRequested.connect(lambda s:self._player.setPosition(int(s*1000)) if self._ap else None)
        self._mp.markerSetRequested.connect(self._on_am_play)
        QShortcut(Qt.Key.Key_Space,self,activated=self._tpp)
        QShortcut(Qt.Key.Key_Escape,self,activated=self._on_stop)
        QShortcut(Qt.Key.Key_L,self,activated=self._on_tls)
        QShortcut(Qt.Key.Key_A,self,activated=self._on_sa)
        QShortcut(Qt.Key.Key_B,self,activated=self._on_sb)
        QShortcut(Qt.Key.Key_T,self,activated=self._on_tap)
        QShortcut(QKeySequence("Ctrl+Z"),self,activated=self._on_undo)
        QShortcut(QKeySequence("Ctrl+Y"),self,activated=self._on_redo)
        QShortcut(QKeySequence("Ctrl+S"),self,activated=self._on_save_p)
        QShortcut(QKeySequence("["),self,activated=lambda:self._skip(-1))
        QShortcut(QKeySequence("]"),self,activated=lambda:self._skip(1))
        QShortcut(QKeySequence("Shift+["),self,activated=lambda:self._skip(-5))
        QShortcut(QKeySequence("Shift+]"),self,activated=lambda:self._skip(5))
        QShortcut(QKeySequence("Ctrl+C"),self,activated=self._on_copy)

    def _style(self):
        self.setStyleSheet("""
            QMainWindow{background:#12121a;}QWidget{background:#12121a;color:#d0d0da;font-family:'Segoe UI','Noto Sans',sans-serif;}
            QPushButton{background:#1e1e2e;border:1px solid #333348;border-radius:6px;padding:3px 10px;color:#d0d0da;font-size:12px;}
            QPushButton:hover{background:#2a2a40;border-color:#5555aa;}QPushButton:pressed{background:#33335a;}
            QPushButton:checked{background:#2e2e50;border-color:#6666cc;color:#aabbff;}
            QPushButton:disabled{background:#16161e;color:#555560;border-color:#222230;}
            QToolBar{background:#16161e;border-bottom:1px solid #222230;spacing:3px;padding:2px;}
            QSlider::groove:horizontal{height:6px;background:#2a2a3a;border-radius:3px;}
            QSlider::handle:horizontal{background:#5577dd;border-radius:6px;width:14px;margin:-4px 0;}
            QSlider::handle:horizontal:hover{background:#6699ff;}
            QSlider::sub-page:horizontal{background:#334488;border-radius:3px;}
            QDoubleSpinBox,QSpinBox{background:#1e1e2e;border:1px solid #333348;border-radius:4px;padding:2px 6px;color:#d0d0da;}
            QStatusBar{background:#0e0e16;color:#888898;font-size:11px;}
            QLabel{background:transparent;}QFrame{background:transparent;border:none;}
            QSplitter::handle{background:#222230;width:2px;}
        """)

    def _load_sett(self):
        if os.path.isfile(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f: s=json.load(f)
                self._rf=s.get("recent_files",[])[:MAX_RECENT]; g=s.get("geometry")
                if g: self.restoreGeometry(bytes.fromhex(g))
            except: pass
    def _save_sett(self):
        try:
            with open(SETTINGS_FILE,'w') as f: json.dump({"recent_files":self._rf[:MAX_RECENT],"geometry":self.saveGeometry().hex()},f,indent=2)
        except: pass
    def _add_rec(self,p):
        p=os.path.abspath(p)
        if p in self._rf: self._rf.remove(p)
        self._rf.insert(0,p); self._rf=self._rf[:MAX_RECENT]; self._build_recent(); self._save_sett()
    def _build_recent(self):
        self._rm.clear()
        if not self._rf: a=self._rm.addAction("(none)"); a.setEnabled(False); return
        for p in self._rf: a=self._rm.addAction(os.path.basename(p)); a.setToolTip(p); a.triggered.connect(lambda c,pa=p:self.load_file(pa))

    def load_file(self,path):
        if not os.path.isfile(path): QMessageBox.warning(self,"Error",f"Not found: {path}"); return
        self._player.stop(); self._pt.stop(); self._lt.stop(); self._lm_w.reset()
        try:
            d,sr=sf.read(path,dtype='float32'); self._ad=d; self._sr=sr; self._ap=path
            self._wf.set_audio(d,sr); self._mm.set_audio(d,sr)
            self._player.setSource(QUrl.fromLocalFile(os.path.abspath(path)))
            for b in[self._b_play,self._b_pause,self._b_stop,self._b_loop,self._b_cl,self._b_sa,self._b_sb2,self._b_sb,self._b_sf,self._ss]: b.setEnabled(True)
            self._b_exp.setEnabled(False); dur=len(d)/sr
            self._lt2.setText(f"0:00.00 / {self._ft(dur)}")
            self._us.clear(); self._rs.clear(); self._mp.clear_markers()
            self._sb.showMessage(f"Loaded: {os.path.basename(path)} | {dur:.2f}s | {sr}Hz -- detecting BPM...")
            QApplication.processEvents()
            bpm=detect_bpm(d,sr); self._wf.set_bpm(bpm)
            self._sb.showMessage(f"Loaded: {os.path.basename(path)} | {dur:.2f}s | {sr}Hz | BPM: {bpm:.1f}" if bpm>0 else f"Loaded: {os.path.basename(path)} | {dur:.2f}s | {sr}Hz | BPM: use Tap/Detect")
            self._add_rec(path)
        except Exception as e: QMessageBox.critical(self,"Error",f"Load failed: {e}")

    def _on_load(self):
        p,_=QFileDialog.getOpenFileName(self,"Open Audio","","Audio (*.wav *.mp3 *.flac *.ogg *.aiff *.aac *.m4a *.wma);;All (*)")
        if p: self.load_file(p)

    def _on_play(self):
        if not self._ap: return
        self._pt.start(); self._lt.start(); self._player.play()
    def _on_pause(self): self._player.pause()
    def _on_stop(self):
        self._player.stop(); self._pt.stop(); self._lt.stop(); self._lm_w.reset()
        self._wf.set_playback_position(0); self._mm.set_playback_position(0)
        self._ss.setValue(0); self._lt2.setText(f"0:00.00 / {self._ft(self._wf._dur)}")
    def _tpp(self):
        if not self._ap: return
        if self._player.playbackState()==QMediaPlayer.PlaybackState.PlayingState: self._on_pause()
        else: self._on_play()
    def _skip(self,d):
        if not self._ap: return
        p=self._player.position()/1000.0+d; p=max(0,min(self._wf._dur,p)); self._player.setPosition(int(p*1000))
    def _on_sc(self,st):
        if st==QMediaPlayer.PlaybackState.PlayingState:
            self._b_play.setText("Playing..."); self._b_play.setStyleSheet("QPushButton{background:#1a3a1a;border-color:#44aa44;color:#66dd66;}")
        elif st==QMediaPlayer.PlaybackState.PausedState:
            self._b_play.setText("Play (paused)"); self._b_play.setStyleSheet("")
        else:
            self._b_play.setText("Play"); self._b_play.setStyleSheet(""); self._lt.stop()

    def _on_pc(self,ms):
        if not self._ap: return; dur=self._player.duration()
        if dur<=0: return
        pos=ms/1000.0; self._wf.set_playback_position(pos); self._mm.set_playback_position(pos)
        if not self._ss.isSliderDown():
            self._ss.blockSignals(True); self._ss.setValue(int(ms/dur*1000)); self._ss.blockSignals(False)
        self._lt2.setText(f"{self._ft(pos)} / {self._ft(dur/1000.0)}")
        if self._looping:
            ls,le=self._wf.get_loop_region()
            if ls>=0 and pos>=le:
                if self._rp.value()>0:
                    self._li+=1
                    if self._li>=self._rp.value(): self._player.stop(); self._pt.stop(); self._lt.stop(); return
                self._player.setPosition(int(ls*1000))
        if self._af:
            vs,ve=self._wf._vs,self._wf._ve
            if pos<vs or pos>ve:
                sp=ve-vs; ns=pos-sp*0.1; ne=ns+sp
                if ns<0: ns=0; ne=sp
                if ne>self._wf._dur: ne=self._wf._dur; ns=ne-sp
                self._wf.set_viewport(max(0,ns),ne); self._mm.set_viewport(max(0,ns),ne)

    def _on_lt_tick(self):
        if self._ad is None: return
        pos=self._player.position()/1000.0; sr=self._sr; fs=int(sr*0.05); cs=int(pos*sr); h=fs//2
        s=max(0,cs-h); e=min(len(self._ad),cs+h); ch=self._ad[s:e]
        if len(ch)==0: self._lm_w.set_levels(0,0,0,0); return
        lc=ch[:,0] if ch.ndim==2 else ch; rc=ch[:,1] if ch.ndim==2 and ch.shape[1]>1 else lc
        self._lm_w.set_levels(float(np.sqrt(np.mean(lc**2))),float(np.sqrt(np.mean(rc**2))),float(np.max(np.abs(lc))),float(np.max(np.abs(rc))))
    def _on_seek_sl(self,v):
        if not self._ap: return; d=self._player.duration()
        if d>0: self._player.setPosition(int(v/1000.0*d))
    def _on_sp(self,s):
        if self._ap: self._player.setPosition(int(s*1000))
    def _on_hov(self,s):
        if self._ad is None: return; s=max(0,min(self._wf._dur,s)); bpm=self._wf._bpm
        info=f"Position: {self._ft(s)}"
        if bpm>0: info+=f"  |  Beat: {s/(60.0/bpm):.2f}"
        if self._wf._snap: info+=f"  |  Snapped: {self._ft(self._wf.snap_to_nearest_beat(s))}"
        self._sb.showMessage(info)
    def _on_mm_nav(self,vs,ve): self._wf.set_viewport(vs,ve)

    def _on_tloop(self):
        self._looping=self._b_loop.isChecked()
        if self._looping:
            ls,le=self._wf.get_loop_region()
            if ls<0: d=self._wf._dur; self._pu(-1,-1); self._wf.set_loop_region(0,d); self._sb.showMessage("Loop ON -- full file")
            else: self._sb.showMessage(f"Loop ON -- {le-ls:.2f}s")
            self._b_loop.setText("Loop: ON"); self._b_loop.setStyleSheet("QPushButton{background:#2e2e50;border-color:#6666cc;color:#aabbff;}")
            self._li=0
        else: self._b_loop.setText("Loop: OFF"); self._b_loop.setStyleSheet(""); self._sb.showMessage("Loop OFF")
    def _on_tls(self): self._b_loop.toggle(); self._on_tloop()

    def _on_sa(self):
        if not self._ad: return
        pos=self._wf.snap_to_nearest_beat(self._player.position()/1000.0); _,ce=self._wf.get_loop_region()
        if ce<0: ce=self._wf._dur
        if pos<ce-0.01: self._pu(self._wf._ls,self._wf._le); self._wf.set_loop_region(pos,ce); self._sb.showMessage(f"A = {self._ft(pos)}")
    def _on_sb(self):
        if not self._ad: return
        pos=self._wf.snap_to_nearest_beat(self._player.position()/1000.0); cs,_=self._wf.get_loop_region()
        if cs<0: cs=0
        if pos>cs+0.01: self._pu(self._wf._ls,self._wf._le); self._wf.set_loop_region(cs,pos); self._sb.showMessage(f"B = {self._ft(pos)}")
    def _on_cl(self):
        self._pu(self._wf._ls,self._wf._le); self._wf.clear_loop_region(); self._b_exp.setEnabled(False)
        self._ll.setText("No loop region -- drag on waveform, or use Set A/B")
        if self._looping: self._b_loop.setChecked(False); self._on_tloop()
    def _on_lc(self,s,e):
        self._wf._markers=self._mp.get_markers()
        if s>=0:
            if e>=0: self._mm.set_loop_region(s,e)
            dur=e-s; bpm=self._wf._bpm; ex=""
            if bpm>0: ex=f"  |  {dur/(60.0/bpm):.1f} beats @ {bpm:.0f} BPM"
            self._ll.setText(f"Loop: A={self._ft(s)} -> B={self._ft(e)} | {self._ft(dur)}{ex}")
            self._b_exp.setEnabled(True); self._li=0; self._pp.update_save_pending(s,e)
        else: self._ll.setText("No loop region -- drag on waveform, or use Set A/B"); self._b_exp.setEnabled(False)

    def _on_nudge(self,which,d):
        if self._wf._ls<0: return
        self._pu(self._wf._ls,self._wf._le)
        if which=="start":
            ns=max(0,self._wf._ls+d)
            if ns<self._wf._le-0.001: self._wf.set_loop_region(ns,self._wf._le)
        elif which=="end":
            ne=min(self._wf._dur,self._wf._le+d)
            if ne>self._wf._ls+0.001: self._wf.set_loop_region(self._wf._ls,ne)

    def _pu(self,s,e):
        self._us.append((s,e))
        if len(self._us)>MAX_UNDO: self._us.pop(0)
        self._rs.clear()
    def _on_undo(self):
        if not self._us: self._sb.showMessage("Nothing to undo"); return
        self._rs.append(self._wf.get_loop_region()); s,e=self._us.pop()
        if s>=0: self._wf.set_loop_region(s,e)
        else: self._wf.clear_loop_region()
        self._sb.showMessage("Undo")
    def _on_redo(self):
        if not self._rs: self._sb.showMessage("Nothing to redo"); return
        self._us.append(self._wf.get_loop_region()); s,e=self._rs.pop()
        if s>=0: self._wf.set_loop_region(s,e)
        else: self._wf.clear_loop_region()
        self._sb.showMessage("Redo")

    def _on_tap(self):
        now=_time.time(); self._tt.append(now); self._tt=self._tt[-8:]
        if len(self._tt)<2: self._sb.showMessage("Tap again..."); return
        ivs=[self._tt[i]-self._tt[i-1] for i in range(1,len(self._tt))]; avg=sum(ivs)/len(ivs)
        bpm=round(60.0/avg,1) if avg>0 else 0
        if 30<=bpm<=300: self._wf.set_bpm(bpm); self._sb.showMessage(f"Tap BPM: {bpm} ({len(ivs)} taps)")
        else: self._sb.showMessage("Out of range, try again")
    def _on_dbpm(self):
        if not self._ad: self._sb.showMessage("Load audio first"); return
        self._sb.showMessage("Detecting BPM..."); QApplication.processEvents()
        bpm=detect_bpm(self._ad,self._sr); self._wf.set_bpm(bpm)
        self._sb.showMessage(f"BPM: {bpm:.1f}" if bpm>0 else "Could not detect -- try Tap")

    def _on_ctx(self,sec):
        sec=max(0,min(self._wf._dur,sec)); menu=QMenu(self)
        menu.setStyleSheet("QMenu{background:#1a1a28;color:#c0c0d0;border:1px solid #333348;}QMenu::item:selected{background:#2a2a50;}")
        a=menu.addAction(f"Set A ({self._ft(sec)})"); a.triggered.connect(lambda:self._sa_at(sec))
        a=menu.addAction(f"Set B ({self._ft(sec)})"); a.triggered.connect(lambda:self._sb_at(sec))
        menu.addSeparator()
        a=menu.addAction("Zoom to Fit"); a.triggered.connect(self._wf.zoom_fit)
        a=menu.addAction("Zoom 10s here"); a.triggered.connect(lambda:self._wf.zoom_to_region(max(0,sec-5),min(self._wf._dur,sec+5)))
        menu.addSeparator()
        a=menu.addAction("Clear Loop"); a.triggered.connect(self._on_cl)
        menu.addSeparator()
        a=menu.addAction(f"Add Marker ({self._ft(sec)})"); a.triggered.connect(lambda:self._on_am(sec))
        a=menu.addAction("Seek here"); a.triggered.connect(lambda:self._on_sp(sec))
        menu.addSeparator()
        a=menu.addAction(f"Copy {sec:.3f}s"); a.triggered.connect(lambda:QApplication.clipboard().setText(f"{sec:.3f}"))
        menu.exec(self._wf.mapToGlobal(self._wf._s2x(int(sec)),40))
    def _sa_at(self,sec):
        if not self._ad: return; _,ce=self._wf.get_loop_region()
        if ce<0: ce=self._wf._dur
        if sec<ce-0.01: self._pu(self._wf._ls,self._wf._le); self._wf.set_loop_region(sec,ce)
    def _sb_at(self,sec):
        if not self._ad: return; cs,_=self._wf.get_loop_region()
        if cs<0: cs=0
        if sec>cs+0.01: self._pu(self._wf._ls,self._wf._le); self._wf.set_loop_region(cs,sec)

    def _on_am(self,sec):
        n,ok=QInputDialog.getText(self,"Add Marker","Name:",text=f"M{len(self._mp.get_markers())+1}")
        if ok and n.strip():
            self._mp.add_marker(n.strip(),sec); self._wf.set_markers(self._mp.get_markers())
            self._sb.showMessage(f"Marker '{n.strip()}' at {self._ft(sec)}")
    def _on_am_play(self):
        if self._ap: self._on_am(self._player.position()/1000.0)

    def _on_norm_loop(self):
        ls,le=self._wf.get_loop_region()
        if ls<0: self._sb.showMessage("Set loop first"); return
        sr=self._sr; s=int(ls*sr); e=int(le*sr); ch=self._ad[s:e]
        pk=np.max(np.abs(ch))
        if pk>0 and pk!=1.0:
            g=1.0/pk; self._ad[s:e]=(self._ad[s:e]*g).astype(np.float32)
            self._wf.set_audio(self._ad,sr); self._mm.set_audio(self._ad,sr)
            self._sb.showMessage(f"Normalized loop by {20*math.log10(1/pk):.1f} dB")
    def _on_norm_all(self):
        if self._ad is None: return
        pk=np.max(np.abs(self._ad))
        if pk>0 and pk!=1.0:
            g=1.0/pk; self._ad=(self._ad*g).astype(np.float32)
            self._wf.set_audio(self._ad,self._sr); self._mm.set_audio(self._ad,self._sr)
            self._sb.showMessage(f"Normalized full audio by {20*math.log10(1/pk):.1f} dB")

    def _on_suggest(self):
        if self._ad is None: self._sb.showMessage("Load audio first"); return
        self._sb.showMessage("Detecting silence..."); QApplication.processEvents()
        regs=detect_silence_regions(self._ad,self._sr)
        if not regs: self._sb.showMessage("No silence found"); return
        if len(regs)>=2:
            ls,le=regs[0][1],regs[1][0]
        elif len(regs)==1:
            ls,le=0.0,regs[0][0]
        else: return
        if le-ls>0.05:
            self._pu(self._wf._ls,self._wf._le); self._wf.set_loop_region(ls,le)
            self._wf.zoom_to_region(ls,le); self._mm.set_viewport(ls,le)
            self._sb.showMessage(f"Suggested: {self._ft(ls)} -> {self._ft(le)} ({le-ls:.2f}s)")
        else: self._sb.showMessage("Region too short")

    def _on_ps(self,s,e):
        self._pu(self._wf._ls,self._wf._le); self._wf.set_loop_region(s,e)
        self._sb.showMessage(f"Preset: {self._ft(s)} -> {self._ft(e)}")
    def _on_save_p(self):
        s,e=self._wf.get_loop_region()
        if s<0: self._sb.showMessage("Set loop first"); return
        self._pp.save_current(s,e); self._pp._add(); self._sb.showMessage(f"Saved preset")

    def _on_export(self):
        if not self._ad: return
        ls,le=self._wf.get_loop_region()
        if ls<0: QMessageBox.information(self,"Export","Set A-B first."); return
        p,_=QFileDialog.getSaveFileName(self,"Export Loop","loop_export.wav","WAV (*.wav);;FLAC (*.flac);;All (*)")
        if not p: return
        sr=self._sr; s=max(0,int(ls*sr)); e=min(len(self._ad),int(le*sr))
        ld=self._ad[s:e].copy(); cf=self._cf.value()
        if cf>0 and len(ld)>0:
            n=int(cf/1000.0*sr)
            if n<len(ld):
                fo=np.linspace(1,0,n,dtype=np.float32); fi=np.linspace(0,1,n,dtype=np.float32)
                if ld.ndim==2: fo=fo[:,np.newaxis]; fi=fi[:,np.newaxis]
                ld[-n:]*=fo; ld[:n]*=fi
        try: sf.write(p,ld,sr); self._sb.showMessage(f"Exported: {p}")
        except Exception as ex: QMessageBox.critical(self,"Error",f"Export failed: {ex}")

    def _on_copy(self):
        ls,le=self._wf.get_loop_region()
        if ls>=0: QApplication.clipboard().setText(f"Loop: {ls:.3f}s - {le:.3f}s ({le-ls:.3f}s)")
        else:
            pos=self._player.position()/1000.0; QApplication.clipboard().setText(f"Position: {pos:.3f}s")
        self._sb.showMessage("Copied to clipboard")

    def _show_keys(self):
        d=QDialog(self); d.setWindowTitle("Keyboard Shortcuts"); d.resize(520,500)
        d.setStyleSheet("QDialog{background:#16161e;}QTextBrowser{background:#12121a;color:#d0d0da;border:1px solid #2a2a40;padding:8px;}")
        la=QVBoxLayout(d)
        tb=QTextBrowser()
        tb.setHtml("""<h3 style="color:#88aaff;">Playback</h3>
<table cellpadding="2"><tr><td><b>Space</b></td><td>Play / Pause</td></tr>
<tr><td><b>Escape</b></td><td>Stop</td></tr>
<tr><td><b>[ / ]</b></td><td>Skip 1s</td></tr>
<tr><td><b>Shift+[ / ]</b></td><td>Skip 5s</td></tr></table>
<h3 style="color:#88aaff;">Loop</h3>
<table cellpadding="2"><tr><td><b>L</b></td><td>Toggle loop</td></tr>
<tr><td><b>A / B</b></td><td>Set loop points</td></tr>
<tr><td><b>Shift+Left/Right</b></td><td>Nudge A by 1ms</td></tr>
<tr><td><b>Alt+Shift+Left/Right</b></td><td>Nudge A by 10ms</td></tr>
<tr><td><b>Ctrl+Shift+Left/Right</b></td><td>Nudge A by 100ms</td></tr>
<tr><td><b>Ctrl+Z / Y</b></td><td>Undo / Redo</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Save preset</td></tr></table>
<h3 style="color:#88aaff;">Navigation</h3>
<table cellpadding="2"><tr><td><b>Left/Right</b></td><td>Scroll</td></tr>
<tr><td><b>+/-</b></td><td>Zoom</td></tr>
<tr><td><b>Home</b></td><td>Zoom fit</td></tr>
<tr><td><b>Ctrl+Scroll</b></td><td>Vertical zoom</td></tr></table>
<h3 style="color:#88aaff;">Other</h3>
<table cellpadding="2"><tr><td><b>T</b></td><td>Tap tempo</td></tr>
<tr><td><b>Ctrl+C</b></td><td>Copy times</td></tr>
<tr><td><b>Right-click</b></td><td>Context menu</td></tr>
<tr><td><b>Dbl-right-click</b></td><td>Add marker</td></tr></table>""")
        la.addWidget(tb)
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Close); bb.accepted.connect(d.accept); bb.rejected.connect(d.reject)
        la.addWidget(bb); d.exec()

    @staticmethod
    def _ft(s):
        if s<0: s=0
        return f"{int(s//60)}:{s%60:05.2f}"
    def closeEvent(self,e): self._save_sett(); self._player.stop(); super().closeEvent(e)


def main():
    app=QApplication(sys.argv); app.setApplicationName("Audio Looper"); app.setApplicationDisplayName("Audio Looper v3")
    w=AudioLooperApp(); w.show(); sys.exit(app.exec())

if __name__=="__main__": main()
