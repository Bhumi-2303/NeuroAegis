import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { HoloBrain } from './HoloBrain';
import { ParticleField } from './ParticleField';

interface SceneProps {
  interactive?: boolean;
}

export function Scene({ interactive = true }: SceneProps) {
  return (
    <div className="w-full h-full relative">
      <Canvas dpr={[1, 2]}>
        <PerspectiveCamera makeDefault position={[0, 0, 10]} fov={45} />
        
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#00E5FF" />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#8B5CF6" />

        <HoloBrain />
        <ParticleField />

        {interactive && (
          <OrbitControls 
            enablePan={false}
            enableZoom={true}
            minDistance={4}
            maxDistance={15}
            autoRotate
            autoRotateSpeed={0.5}
          />
        )}

        <EffectComposer>
          <Bloom 
            luminanceThreshold={0.2} 
            mipmapBlur 
            intensity={1.5} 
            radius={0.4}
          />
        </EffectComposer>
      </Canvas>
    </div>
  );
}
