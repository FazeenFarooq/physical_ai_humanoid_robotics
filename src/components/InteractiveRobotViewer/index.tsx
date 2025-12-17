import type {ReactNode} from 'react';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

export default function InteractiveRobotViewer(): ReactNode {
  return (
    <section className={styles.robotViewerContainer}>
      <div className="container">
        <Heading as="h2" className={styles.robotViewerHeader}>
          Interactive Robot Model
        </Heading>
        <div className={styles.robotViewer}>
          <div className={styles.robotModelPlaceholder}>
            <p>3D Robot Model Viewer</p>
            <p>(Will be implemented with Three.js)</p>
            <p>Rotate • Zoom • Explore Components</p>
          </div>
        </div>
      </div>
    </section>
  );
}