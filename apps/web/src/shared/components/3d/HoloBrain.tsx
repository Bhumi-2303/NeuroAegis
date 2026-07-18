import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface HoloBrainProps {
  radius?: number;
  nodeCount?: number;
}

export function HoloBrain({ radius = 4, nodeCount = 400 }: HoloBrainProps) {
  const groupRef = useRef<THREE.Group>(null);

  // Generate points on an ellipsoid
  const { positions, lines } = useMemo(() => {
    const points: THREE.Vector3[] = [];
    const a = radius;
    const b = radius * 0.7; // Squashed vertically
    const c = radius * 1.2; // Elongated front-to-back

    for (let i = 0; i < nodeCount; i++) {
      // Random point in sphere
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);
      
      // Push out to surface, add some noise
      const r = 0.8 + Math.random() * 0.2;
      const x = r * a * Math.sin(phi) * Math.cos(theta);
      const y = r * b * Math.sin(phi) * Math.sin(theta);
      const z = r * c * Math.cos(phi);

      // Simple hemispheres (left/right split)
      const isLeft = x < 0;
      const gap = 0.2;
      const adjustedX = isLeft ? x - gap : x + gap;

      points.push(new THREE.Vector3(adjustedX, y, z));
    }

    // Connect nearby points to form "synapses"
    const linePoints: THREE.Vector3[] = [];
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        if (points[i].distanceTo(points[j]) < 1.5) {
          if (Math.random() > 0.5) { // Only connect some
            linePoints.push(points[i], points[j]);
          }
        }
      }
    }

    const posArray = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {
      posArray[i * 3] = points[i].x;
      posArray[i * 3 + 1] = points[i].y;
      posArray[i * 3 + 2] = points[i].z;
    }

    return { positions: posArray, lines: linePoints };
  }, [radius, nodeCount]);

  useFrame((_state, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.1;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Outer Shell */}
      <mesh>
        <sphereGeometry args={[radius * 1.3, 32, 32]} />
        <meshBasicMaterial 
          color="#8B5CF6" 
          wireframe 
          transparent 
          opacity={0.03} 
        />
      </mesh>

      {/* Nodes */}
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[positions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.05}
          color="#00E5FF"
          transparent
          opacity={0.8}
          sizeAttenuation
        />
      </points>

      {/* Synapses */}
      {lines.length > 0 && (
        <lineSegments>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              args={[new Float32Array(lines.flatMap(v => [v.x, v.y, v.z])), 3]}
            />
          </bufferGeometry>
          <lineBasicMaterial
            color="#4B7DFF"
            transparent
            opacity={0.15}
          />
        </lineSegments>
      )}

      {/* Holographic Rings */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[radius * 1.5, radius * 1.52, 64]} />
        <meshBasicMaterial color="#00E5FF" transparent opacity={0.2} side={THREE.DoubleSide} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, -radius*0.5, 0]}>
        <ringGeometry args={[radius * 1.2, radius * 1.21, 64]} />
        <meshBasicMaterial color="#8B5CF6" transparent opacity={0.1} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}
