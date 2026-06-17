/* math/fourier-dct 笔记的交互 demo（浅色主题：白底画布 + 深色文字/线条）。
 * navigation.instant 下内联 <script> 翻页不重跑，故走 extra_javascript + document$.subscribe，
 * 并对每个 demo 做存在性守卫。 */
(function () {
  function run() {
    const $ = (id) => document.getElementById(id);
    if (!$('cvSquare') && !$('cvBasis') && !$('cvCompact')) return;

    const INK='#201F1C', MUTED='#6B675F', LINE='#D8D2C7', ZERO='#E4E0D7', REF='#8C8780';
    const BLUE='#58a6ff', GREEN='#3fb950', RED='#f85149', PURPLE='#bc8cff';

    function dctO(x){const N=x.length,X=new Array(N).fill(0);for(let k=0;k<N;k++){let s=0;for(let n=0;n<N;n++)s+=x[n]*Math.cos(Math.PI/N*(n+0.5)*k);X[k]=(k===0?Math.sqrt(1/N):Math.sqrt(2/N))*s;}return X;}
    function idctO(X){const N=X.length,x=new Array(N).fill(0);for(let n=0;n<N;n++){let s=0;for(let k=0;k<N;k++){const f=k===0?Math.sqrt(1/N):Math.sqrt(2/N);s+=f*X[k]*Math.cos(Math.PI/N*(n+0.5)*k);}x[n]=s;}return x;}

    /* ===== 方波傅里叶级数 ===== */
    (function(){
      const cv=$('cvSquare'); if(!cv) return; const x=cv.getContext('2d'),N=$('sqN'),NV=$('sqNV'),out=$('sqOut');
      function draw(){
        const W=cv.width,H=cv.height,mid=H/2,amp=H*0.32,nh=+N.value;NV.textContent=nh;x.clearRect(0,0,W,H);
        x.strokeStyle=LINE;x.beginPath();x.moveTo(20,mid);x.lineTo(W-10,mid);x.stroke();
        x.strokeStyle=REF;x.lineWidth=2;x.beginPath();
        const M=900;for(let i=0;i<M;i++){const t=i/M;const sq=(t%1)<0.5?1:-1;const px=20+t*(W-30),py=mid-sq*amp;i?x.lineTo(px,py):x.moveTo(px,py);}x.stroke();
        x.strokeStyle=RED;x.lineWidth=2;x.beginPath();
        for(let i=0;i<M;i++){const t=i/M;let v=0;for(let h=1;h<=2*nh-1;h+=2)v+=(4/Math.PI)*Math.sin(2*Math.PI*h*t)/h;const px=20+t*(W-30),py=mid-v*amp;i?x.lineTo(px,py):x.moveTo(px,py);}x.stroke();
        x.fillStyle=REF;x.font='12px sans-serif';x.fillText('目标方波',26,20);x.fillStyle=RED;x.fillText('前 '+nh+' 个奇次谐波之和',100,20);
        out.textContent=`用了 ${nh} 个正弦谐波（频率 1,3,5,…,${2*nh-1}）。注意跳变处的过冲（吉布斯现象）始终存在 ≈9%。`;
      }
      N.oninput=draw;draw();
    })();

    /* ===== DCT 基函数画廊 ===== */
    (function(){
      const cv=$('cvBasis'); if(!cv) return; const x=cv.getContext('2d'),Nn=32;let hi=-1;
      function basis(k){const v=[];for(let n=0;n<Nn;n++)v.push(Math.cos(Math.PI/Nn*(n+0.5)*k));return v;}
      function draw(){
        const W=cv.width,H=cv.height;x.clearRect(0,0,W,H);const cols=4,rows=2,cw=W/cols,ch=H/rows;
        for(let k=0;k<8;k++){const cx=(k%cols)*cw,cy=Math.floor(k/cols)*ch,midY=cy+ch/2;
          x.fillStyle=k===hi?'rgba(88,166,255,.10)':'transparent';x.fillRect(cx+2,cy+2,cw-4,ch-4);
          x.strokeStyle=k===hi?BLUE:LINE;x.strokeRect(cx+4,cy+4,cw-8,ch-8);
          x.strokeStyle=LINE;x.beginPath();x.moveTo(cx+16,midY);x.lineTo(cx+cw-12,midY);x.stroke();
          const v=basis(k);x.strokeStyle=k===0?GREEN:PURPLE;x.lineWidth=2;x.beginPath();
          for(let n=0;n<Nn;n++){const px=cx+16+n/(Nn-1)*(cw-28),py=midY-v[n]*(ch*0.32);n?x.lineTo(px,py):x.moveTo(px,py);}x.stroke();
          x.fillStyle=MUTED;x.font='12px sans-serif';x.fillText('k='+k+(k===0?' (直流/均值)':''),cx+16,cy+22);}
      }
      cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect();const sx=cv.width/r.width,sy=cv.height/r.height;
        const px=(e.clientX-r.left)*sx,py=(e.clientY-r.top)*sy;
        const cw=cv.width/4,ch=cv.height/2;hi=Math.floor(py/ch)*4+Math.floor(px/cw);draw();});
      cv.addEventListener('mouseleave',()=>{hi=-1;draw();});
      draw();
    })();

    /* ===== 能量压缩 ===== */
    (function(){
      const cv=$('cvCompact'); if(!cv) return; const x=cv.getContext('2d'),N=32;
      const K=$('ckK'),KV=$('ckKV'),out=$('ckOut');
      const bS=$('ckSmooth'),bR=$('ckRough');let rough=false;
      function sig(){const a=[];for(let n=0;n<N;n++){const t=n/(N-1);
        if(!rough)a.push(0.7*Math.sin(2*Math.PI*t+0.5)+0.3*Math.sin(2*Math.PI*2*t));
        else{a.push(0.5*Math.sin(2*Math.PI*t)+0.4*Math.sin(2*Math.PI*7*t)+0.3*Math.sin(2*Math.PI*13*t));}}return a;}
      function draw(){
        const W=cv.width,H=cv.height,k=+K.value;KV.textContent=k;x.clearRect(0,0,W,H);
        const A=sig(),C=dctO(A),Ck=C.map((c,i)=>i<k?c:0),rec=idctO(Ck);
        const mid=110,amp=80;x.fillStyle=MUTED;x.font='12px sans-serif';x.fillText('原始信号（蓝） vs 用前 k 个系数重建（红）',14,20);
        x.strokeStyle=LINE;x.beginPath();x.moveTo(20,mid);x.lineTo(W-10,mid);x.stroke();
        x.strokeStyle=BLUE;x.lineWidth=2.5;x.beginPath();for(let n=0;n<N;n++){const px=20+n/(N-1)*(W-30),py=mid-A[n]*amp;n?x.lineTo(px,py):x.moveTo(px,py);}x.stroke();
        x.strokeStyle=RED;x.lineWidth=2;x.setLineDash([5,3]);x.beginPath();for(let n=0;n<N;n++){const px=20+n/(N-1)*(W-30),py=mid-rec[n]*amp;n?x.lineTo(px,py):x.moveTo(px,py);}x.stroke();x.setLineDash([]);
        const by=320,maxC=Math.max(...C.map(Math.abs),1e-6),bw=(W-40)/N;
        x.fillStyle=MUTED;x.fillText('DCT 系数（保留的高亮，丢弃的灰）：',14,210);
        let kept=0,tot=0;for(let i=0;i<N;i++){tot+=C[i]*C[i];if(i<k)kept+=C[i]*C[i];}
        for(let i=0;i<N;i++){const h=Math.abs(C[i])/maxC*90,px=20+i*bw;x.fillStyle=i<k?PURPLE:ZERO;x.fillRect(px,by-h,bw-2,h);}
        x.strokeStyle=LINE;x.beginPath();x.moveTo(20,by);x.lineTo(W-10,by);x.stroke();
        const mse=A.reduce((s,a,i)=>s+(a-rec[i])**2,0)/N;
        out.textContent=`${rough?'含高频/噪声':'平滑'}信号 ｜ 保留前 ${k}/${N} 个系数 ｜ 捕获能量 ${(kept/tot*100).toFixed(1)}% ｜ 重建 MSE = ${mse.toExponential(2)}。${rough?'高频信号需要更多系数才能重建好。':'平滑信号只需极少系数即几乎重合——这就是能量压缩。'}`;
      }
      K.oninput=draw;bS.onclick=()=>{rough=false;bS.classList.add('on');bR.classList.remove('on');draw();};
      bR.onclick=()=>{rough=true;bR.classList.add('on');bS.classList.remove('on');draw();};
      draw();
    })();
  }

  if (typeof window.document$ !== 'undefined' && window.document$.subscribe) {
    window.document$.subscribe(run);
  } else if (document.readyState !== 'loading') {
    run();
  } else {
    document.addEventListener('DOMContentLoaded', run);
  }
})();
