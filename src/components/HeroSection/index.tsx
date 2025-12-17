import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

export default function HeroSection(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.bookCoverContainer}>
            <div className={styles.bookCover}>
              <div className={styles.bookSpine}></div>
              <div className={styles.bookFront}>
                <h1 className={styles.heroTitle}>AI-Driven Physical Robotics</h1>
                <p className={styles.heroSubtitle}>Creating Humanoid Robots with Advanced AI Systems</p>
              </div>
            </div>
          </div>
          <div className={styles.heroText}>
            <Heading as="h1" className={styles.mainTitle}>
              AI-Driven Physical Robotics
            </Heading>
            <p className={styles.heroSubtitle}>
              Explore the cutting-edge field of AI-powered humanoid development and discover how neural networks bring metal to life.
            </p>
            <div className={styles.buttons}>
              <Link className={styles.primaryButton} to="/docs/intro">
                Get the Book
              </Link>
              <Link className={styles.secondaryButton} to="/docs/intro">
                Watch Trailer
              </Link>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}