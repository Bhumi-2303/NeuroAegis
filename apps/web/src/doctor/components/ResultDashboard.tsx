import React, { useEffect, useState, useRef } from 'react';
import { Download, Printer, AlertTriangle, ShieldCheck, Activity, Brain } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';

// --- 3D Brain Network Component ---
function BrainNetwork() {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame(({ clock }) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = clock.getElapsedTime() * 0.1;
    }
  });

  // Generate some random nodes in a roughly spherical shape to simulate a brain
  const nodes = React.useMemo(() => {
    const pts = [];
    for (let i = 0; i < 50; i++) {
      const phi = Math.acos(-1 + (2 * i) / 50);
      const theta = Math.sqrt(50 * Math.PI) * phi;
      const x = 2.5 * Math.cos(theta) * Math.sin(phi);
      const y = 2 * Math.cos(phi);
      const z = 2.5 * Math.sin(theta) * Math.sin(phi);
      pts.push(new THREE.Vector3(x, y, z));
    }
    return pts;
  }, []);

  const lines = React.useMemo(() => {
    const lns = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        if (nodes[i].distanceTo(nodes[j]) < 1.5) {
          lns.push([nodes[i], nodes[j]]);
        }
      }
    }
    return lns;
  }, [nodes]);

  return (
    <group ref={groupRef}>
      {nodes.map((pos, i) => (
        <Sphere key={i} position={pos} args={[0.08, 16, 16]}>
          <meshBasicMaterial color={i % 5 === 0 ? "#f43f5e" : "#2dd4bf"} />
        </Sphere>
      ))}
      {lines.map((line, i) => (
        <Line key={i} points={line} color="#3b82f6" opacity={0.2} transparent lineWidth={1} />
      ))}
    </group>
  );
}

// --- Main Result Dashboard ---
export function ResultDashboard({ jobId }: { jobId: string }) {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    const fetchReport = async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/v2/predict/status/${jobId}`);
        const result = await res.json();
        setData(result.result);
      } catch (e) {
        console.error(e);
      }
    };
    fetchReport();
  }, [jobId]);

  if (!data) {
    return <div className="h-96 flex items-center justify-center">Loading results...</div>;
  }

  const isHighRisk = data.prediction_label === 'seizure';
  const prob = (data.probability_seizure * 100).toFixed(1);
  const confidence = data.confidence_band.toUpperCase();
  
  // Format SHAP data for Recharts
  const shapData = Object.entries(data.shap_explanation).map(([name, value]) => ({
    name,
    value: Number(value)
  })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 8); // top 8

  return (
    <div className="space-y-6">
      {/* Header Actions */}
      <div className="flex justify-end gap-4 mb-4">
        <button className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm transition-colors border border-white/10">
          <Download className="w-4 h-4" /> PDF Report
        </button>
        <button className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm transition-colors border border-white/10">
          <Printer className="w-4 h-4" /> Print
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Main Prediction Card */}
        <div className={`lg:col-span-2 relative overflow-hidden rounded-2xl border ${isHighRisk ? 'border-red-500/30 bg-red-500/5' : 'border-teal-500/30 bg-teal-500/5'} p-8 backdrop-blur-xl`}>
          <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-white/5 to-transparent rounded-full blur-3xl -translate-y-1/2 translate-x-1/3" />
          
          <div className="flex items-start justify-between relative z-10">
            <div>
              <div className="flex items-center gap-3 mb-2">
                {isHighRisk ? <AlertTriangle className="w-8 h-8 text-red-400" /> : <ShieldCheck className="w-8 h-8 text-teal-400" />}
                <h2 className="text-3xl font-bold tracking-tight">AI Clinical Assessment</h2>
              </div>
              <p className="text-neutral-400 max-w-md mt-4">
                Based on the GNN analysis of the uploaded EEG signals, the neural activity patterns exhibit characteristics consistent with {isHighRisk ? 'abnormal epileptiform activity' : 'normal baseline rhythms'}.
              </p>
            </div>
            
            <div className="text-right">
              <p className="text-sm font-medium text-neutral-400 mb-1">Seizure Probability</p>
              <div className="text-6xl font-light tracking-tighter">
                <span className={isHighRisk ? 'text-red-400' : 'text-teal-400'}>{prob}%</span>
              </div>
              <div className="mt-4 flex items-center justify-end gap-2 text-sm">
                <span className="text-neutral-500">Confidence Model:</span>
                <span className={`px-2 py-1 rounded text-xs font-bold ${
                  confidence === 'HIGH' ? 'bg-teal-500/20 text-teal-300' : 
                  confidence === 'MEDIUM' ? 'bg-yellow-500/20 text-yellow-300' : 'bg-red-500/20 text-red-300'
                }`}>{confidence}</span>
              </div>
            </div>
          </div>
        </div>

        {/* 3D Brain Viz */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-4 backdrop-blur-xl flex flex-col h-64 lg:h-auto relative overflow-hidden">
          <div className="absolute top-4 left-4 z-10 flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" />
            <span className="text-sm font-medium">Neural Connectivity</span>
          </div>
          <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-neutral-950/80 to-transparent z-10" />
          <Canvas camera={{ position: [0, 0, 8], fov: 45 }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1} />
            <BrainNetwork />
            <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={1} />
          </Canvas>
        </div>

        {/* SHAP Feature Importance */}
        <div className="lg:col-span-3 bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-xl">
          <div className="flex items-center gap-3 mb-6 pb-4 border-b border-white/10">
            <Activity className="w-5 h-5 text-blue-400" />
            <h3 className="text-xl font-medium">Feature Contribution (SHAP)</h3>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={shapData} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fill: '#9ca3af', fontSize: 12 }} width={120} />
                <Tooltip 
                  cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                  contentStyle={{ backgroundColor: '#171717', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {shapData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#ef4444' : '#3b82f6'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-6 mt-4 text-xs text-neutral-400">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500" /> Increased Risk
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-500" /> Decreased Risk
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
