import type {ReactNode} from 'react';
import styles from './styles.module.css';

type StatusItem = {
  number: string;
  label: string;
};

const StatusItems: StatusItem[] = [
  { number: '15+', label: 'Active Projects' },
  { number: '420+', label: 'AI Models Trained' },
  { number: '8+', label: 'Robots in Development' },
  { number: '99%', label: 'Accuracy Rate' },
];

export default function StatusPanel(): ReactNode {
  return (
    <div className={styles.statusPanel}>
      {StatusItems.map((item, index) => (
        <div key={index} className={styles.statusItem}>
          <div className={styles.statusNumber}>{item.number}</div>
          <div className={styles.statusLabel}>{item.label}</div>
        </div>
      ))}
    </div>
  );
}