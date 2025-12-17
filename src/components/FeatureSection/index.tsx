import type {ReactNode} from 'react';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  description: string;
  imageUrl: string;
  buttonText: string;
  link: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'AI Experiments',
    description: 'Explore cutting-edge AI research in robotics with our collection of experiments and implementations.',
    imageUrl: '/img/ai-experiment.jpg',
    buttonText: 'Explore Experiments',
    link: '/docs/ai-experiments',
  },
  {
    title: 'Robotics Prototypes',
    description: 'View our latest humanoid robotics prototypes and their technical specifications.',
    imageUrl: '/img/robot-prototype.jpg',
    buttonText: 'View Prototypes',
    link: '/docs/prototypes',
  },
  {
    title: 'Project Updates',
    description: 'Stay updated with the latest developments and progress in our humanoid robotics project.',
    imageUrl: '/img/project-updates.jpg',
    buttonText: 'View Updates',
    link: '/blog',
  },
];

function Feature({title, description, imageUrl, buttonText, link}: FeatureItem): ReactNode {
  return (
    <div className={styles.featureCard}>
      <img 
        src={imageUrl} 
        alt={title} 
        className={styles.featureImage}
        onError={(e) => {
          const target = e.target as HTMLImageElement;
          target.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 24 24"><rect width="24" height="24" fill="%23121220"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-size="12" fill="%2300D4FF">Robot Image</text></svg>';
        }}
      />
      <h3 className={styles.featureTitle}>{title}</h3>
      <p className={styles.featureDescription}>{description}</p>
      <a href={link} className={styles.featureButton}>
        {buttonText}
      </a>
    </div>
  );
}

export default function FeatureSection(): ReactNode {
  return (
    <section className={styles.sectionContainer}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">Explore the Future of Robotics</Heading>
        </div>
        <div className={styles.featureGrid}>
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}