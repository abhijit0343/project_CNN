// components/BrainNetwork.jsx
import React, { useEffect, useRef, useState } from 'react';

export default function BrainNetwork() {
  const canvasRef = useRef(null);
  const [mousePos, setMousePos] = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId;
    let width = (canvas.width = canvas.parentElement?.clientWidth || 600);
    let height = (canvas.height = 360);

    const handleResize = () => {
      if (!canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = 360;
    };
    window.addEventListener('resize', handleResize);

    const points = [];
    const numPoints = 64;

    const centerX = width / 2;
    const centerY = height / 2;

    for (let i = 0; i < numPoints; i++) {
      const isLeft = i % 2 === 0;
      const angle = (i / numPoints) * Math.PI * 2;
      const rx = 180 * (0.6 + 0.4 * Math.random());
      const ry = 110 * (0.6 + 0.4 * Math.random());
      
      const lobeShift = isLeft ? -45 : 45;
      const ox = centerX + lobeShift + Math.cos(angle) * (rx * 0.7);
      const oy = centerY + Math.sin(angle) * (ry * 0.8) + (Math.sin(angle * 2) * 20);

      const isRed = i % 7 === 0 || i === 3 || i === 19;
      points.push({
        x: ox,
        y: oy,
        originX: ox,
        originY: oy,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        isRed,
        size: isRed ? 3.2 : 1.8 + Math.random() * 1.2,
        pulsePhase: Math.random() * Math.PI * 2,
      });
    }

    const pulses = [];
    for (let p = 0; p < 8; p++) {
      pulses.push({
        fromIndex: Math.floor(Math.random() * points.length),
        toIndex: Math.floor(Math.random() * points.length),
        progress: Math.random(),
        speed: 0.006 + Math.random() * 0.008,
      });
    }

    let time = 0;

    const render = () => {
      time += 0.02;
      ctx.clearRect(0, 0, width, height);

      const grad = ctx.createRadialGradient(centerX, centerY, 30, centerX, centerY, 240);
      grad.addColorStop(0, 'rgba(66, 102, 121, 0.12)');
      grad.addColorStop(1, 'rgba(0, 49, 83, 0)');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, width, height);

      ctx.beginPath();
      ctx.setLineDash([4, 6]);
      ctx.strokeStyle = 'rgba(189, 210, 223, 0.18)';
      ctx.lineWidth = 1;
      ctx.moveTo(centerX, centerY - 120);
      ctx.lineTo(centerX, centerY + 120);
      ctx.stroke();
      ctx.setLineDash([]);

      points.forEach((pt) => {
        pt.x += pt.vx;
        pt.y += pt.vy;
        if (Math.abs(pt.x - pt.originX) > 12) pt.vx *= -1;
        if (Math.abs(pt.y - pt.originY) > 12) pt.vy *= -1;

        if (mousePos) {
          const dx = mousePos.x - pt.x;
          const dy = mousePos.y - pt.y;
          const dist = Math.hypot(dx, dy);
          if (dist < 90) {
            pt.x -= (dx / dist) * 1.5;
            pt.y -= (dy / dist) * 1.5;
          }
        }
      });

      for (let i = 0; i < points.length; i++) {
        for (let j = i + 1; j < points.length; j++) {
          const a = points[i];
          const b = points[j];
          const dist = Math.hypot(a.x - b.x, a.y - b.y);

          if (dist < 72) {
            const alpha = (1 - dist / 72) * 0.28;
            ctx.beginPath();
            if (a.isRed && b.isRed) {
              ctx.strokeStyle = `rgba(214, 40, 57, ${alpha * 1.4})`;
              ctx.lineWidth = 1.1;
            } else if (a.isRed || b.isRed) {
              ctx.strokeStyle = `rgba(189, 210, 223, ${alpha * 1.1})`;
              ctx.lineWidth = 0.8;
            } else {
              ctx.strokeStyle = `rgba(183, 207, 221, ${alpha * 0.7})`;
              ctx.lineWidth = 0.6;
            }
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      pulses.forEach((pulse) => {
        pulse.progress += pulse.speed;
        if (pulse.progress >= 1) {
          pulse.progress = 0;
          pulse.fromIndex = Math.floor(Math.random() * points.length);
          const from = points[pulse.fromIndex];
          const neighbors = points
            .map((pt, idx) => ({ idx, dist: Math.hypot(pt.x - from.x, pt.y - from.y) }))
            .filter((n) => n.dist > 10 && n.dist < 85);
          if (neighbors.length > 0) {
            pulse.toIndex = neighbors[Math.floor(Math.random() * neighbors.length)].idx;
          } else {
            pulse.toIndex = (pulse.fromIndex + 1) % points.length;
          }
        }

        const pA = points[pulse.fromIndex];
        const pB = points[pulse.toIndex];
        if (pA && pB) {
          const px = pA.x + (pB.x - pA.x) * pulse.progress;
          const py = pA.y + (pB.y - pA.y) * pulse.progress;

          ctx.beginPath();
          ctx.arc(px, py, 1.8, 0, Math.PI * 2);
          ctx.fillStyle = '#D62839';
          ctx.shadowColor = '#D62839';
          ctx.shadowBlur = 6;
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      });

      points.forEach((pt) => {
        const pulseVal = Math.sin(time * 2 + pt.pulsePhase);
        ctx.beginPath();

        if (pt.isRed) {
          const rad = pt.size + pulseVal * 0.8;
          ctx.arc(pt.x, pt.y, Math.max(1, rad), 0, Math.PI * 2);
          ctx.fillStyle = '#D62839';
          ctx.shadowColor = 'rgba(214, 40, 57, 0.7)';
          ctx.shadowBlur = 8;
          ctx.fill();
          ctx.shadowBlur = 0;

          ctx.beginPath();
          ctx.arc(pt.x, pt.y, rad + 4 + pulseVal * 1.5, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(214, 40, 57, ${0.35 + pulseVal * 0.15})`;
          ctx.lineWidth = 0.8;
          ctx.stroke();
        } else {
          ctx.arc(pt.x, pt.y, pt.size, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(215, 231, 241, 0.82)';
          ctx.fill();
        }
      });

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
    };
  }, [mousePos]);

  return (
    <div
      className="relative w-full h-[360px] flex items-center justify-center select-none"
      onMouseMove={(e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
      }}
      onMouseLeave={() => setMousePos(null)}
    >
      <canvas
        ref={canvasRef}
        className="w-full h-full block"
        aria-label="Interactive 60fps brain neural network visualization"
      />
    </div>
  );
}
