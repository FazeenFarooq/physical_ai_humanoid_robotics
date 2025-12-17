import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

// Import the new UI components
import HeroSection from '@site/src/components/HeroSection';
import FeatureSection from '@site/src/components/FeatureSection';
import InteractiveRobotViewer from '@site/src/components/InteractiveRobotViewer';
import StatusPanel from '@site/src/components/StatusPanel';

function NeuralNetworkBackground() {
  return (
    <div className={styles.neuralNetworkBackground}>
      <svg className={styles.neuralNetworkSvg} viewBox="0 0 100 100" preserveAspectRatio="none">
        <path d="M10,10 C20,30 30,20 40,30 C50,40 60,30 70,50 C80,70 90,60 90,60" stroke="var(--ifm-color-primary)" strokeWidth="0.1" fill="none" opacity="0.3" />
        <path d="M15,85 C25,65 35,75 45,60 C55,45 65,55 75,40 C85,25 85,25 85,25" stroke="var(--ifm-color-primary)" strokeWidth="0.1" fill="none" opacity="0.3" />
        <path d="M5,50 C15,40 25,45 35,35 C45,25 55,30 65,40 C75,50 85,45 95,50" stroke="var(--ifm-color-primary)" strokeWidth="0.1" fill="none" opacity="0.3" />
        <circle cx="10" cy="10" r="0.5" fill="var(--ifm-color-primary)" />
        <circle cx="40" cy="30" r="0.5" fill="var(--ifm-color-primary)" />
        <circle cx="70" cy="50" r="0.5" fill="var(--ifm-color-primary)" />
        <circle cx="90" cy="60" r="0.5" fill="var(--ifm-color-primary)" />
        <circle cx="15" cy="85" r="0.5" fill="var(--ifm-color-primary)" />
        <circle cx="75" cy="40" r="0.5" fill="var(--ifm-color-primary)" />
      </svg>
    </div>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`AI-Driven Physical Robotics`}
      description="Discover how to create humanoid robots with advanced AI systems">
      <NeuralNetworkBackground />
      <HeroSection />
      <main>
        <StatusPanel />
        <FeatureSection />
        <InteractiveRobotViewer />
      </main>
    </Layout>
  );
}
