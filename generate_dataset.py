import csv
import numpy as np
import os

LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
          'OK', 'Stop', 'thankyou', 'help', 'i love you', 'hello', 'sorry', 'yes', 'no']
SAMPLES_PER_CLASS = 1500

def get_base_template(label):
    landmarks = np.zeros((21, 2))
    
    # Define finger states
    def set_thumb(state):
        if state == 'up': landmarks[1:5] = [[-10, -20], [-10, -40], [-10, -60], [-10, -80]]
        elif state == 'out': landmarks[1:5] = [[-20, -10], [-40, -10], [-60, -10], [-80, -10]]
        elif state == 'across': landmarks[1:5] = [[-10, -10], [10, -15], [30, -15], [40, -15]]
        elif state == 'touch_index': landmarks[1:5] = [[-20, -15], [-35, -30], [-45, -50], [-40, -70]]
        elif state == 'tucked': landmarks[1:5] = [[-10, -10], [0, -15], [10, -20], [15, -20]]
        elif state == 'touch_middle': landmarks[1:5] = [[-15, -15], [-25, -30], [-30, -45], [-20, -60]]

    def set_finger(idx, base_x, base_y, length_mult, state):
        base = [base_x, base_y]
        pts = [base]
        if state == 'straight':
            pts.extend([[base_x, base_y - 30*length_mult], [base_x, base_y - 60*length_mult], [base_x, base_y - 80*length_mult]])
        elif state == 'curled':
            pts.extend([[base_x, base_y + 15], [base_x, base_y + 30], [base_x, base_y + 40]])
        elif state == 'curved':
            pts.extend([[base_x - 10, base_y - 20], [base_x - 30, base_y - 30], [base_x - 40, base_y - 20]])
        elif state == 'straight_spread_left':
            pts.extend([[base_x - 15, base_y - 30*length_mult], [base_x - 30, base_y - 60*length_mult], [base_x - 45, base_y - 80*length_mult]])
        elif state == 'straight_spread_right':
            pts.extend([[base_x + 15, base_y - 30*length_mult], [base_x + 30, base_y - 60*length_mult], [base_x + 45, base_y - 80*length_mult]])
        elif state == 'cross_left':
            pts.extend([[base_x - 10, base_y - 30*length_mult], [base_x - 20, base_y - 60*length_mult], [base_x - 30, base_y - 80*length_mult]])
        elif state == 'cross_right':
            pts.extend([[base_x + 10, base_y - 30*length_mult], [base_x + 20, base_y - 60*length_mult], [base_x + 30, base_y - 80*length_mult]])
        landmarks[idx:idx+4] = pts

    def set_index(s): set_finger(5, -15, -45, 1.0, s)
    def set_middle(s): set_finger(9, 0, -50, 1.1, s)
    def set_ring(s): set_finger(13, 15, -45, 1.0, s)
    def set_pinky(s): set_finger(17, 30, -35, 0.8, s)

    # Defaults
    t_s, i_s, m_s, r_s, p_s = 'tucked', 'curled', 'curled', 'curled', 'curled'

    # ASL Dictionary mapping
    if label in ['A', 'sorry']: t_s = 'up'; i_s = 'curled'; m_s = 'curled'; r_s = 'curled'; p_s = 'curled'
    elif label in ['B', 'thankyou', 'hello']: t_s = 'across'; i_s = 'straight'; m_s = 'straight'; r_s = 'straight'; p_s = 'straight'
    elif label == 'C': t_s = 'touch_index'; i_s = 'curved'; m_s = 'curved'; r_s = 'curved'; p_s = 'curved'
    elif label == 'D': t_s = 'touch_middle'; i_s = 'straight'; m_s = 'curled'; r_s = 'curled'; p_s = 'curled'
    elif label == 'E': t_s = 'across'; i_s = 'curled'; m_s = 'curled'; r_s = 'curled'; p_s = 'curled'
    elif label in ['F', 'OK']: t_s = 'touch_index'; i_s = 'curved'; m_s = 'straight_spread_right'; r_s = 'straight_spread_right'; p_s = 'straight_spread_right'
    elif label == 'G': t_s = 'out'; i_s = 'straight_spread_right'; m_s = 'curled'; r_s = 'curled'; p_s = 'curled'
    elif label == 'H': t_s = 'tucked'; i_s = 'straight_spread_right'; m_s = 'straight_spread_right'; r_s = 'curled'; p_s = 'curled'
    elif label in ['I', 'J']: t_s = 'across'; i_s = 'curled'; m_s = 'curled'; r_s = 'curled'; p_s = 'straight'
    elif label == 'K': t_s = 'up'; i_s = 'straight'; m_s = 'straight_spread_right'; r_s = 'curled'; p_s = 'curled'
    elif label == 'L': t_s = 'out'; i_s = 'straight'; m_s = 'curled'; r_s = 'curled'; p_s = 'curled'
    elif label == 'M': t_s = 'tucked'; i_s = 'curled'; m_s = 'curled'; r_s = 'curled'; p_s = 'curled'
    elif label == 'N': t_s = 'tucked'; i_s = 'curled'; m_s = 'curled'; r_s = 'straight'; p_s = 'straight'
    elif label == 'O': t_s = 'touch_index'; i_s = 'curved'; m_s = 'curved'; r_s = 'curved'; p_s = 'curved'
    elif label == 'P': t_s = 'out'; i_s = 'straight_spread_right'; m_s = 'straight_spread_left'; r_s = 'curled'; p_s = 'curled'
    elif label == 'Q': t_s = 'out'; i_s = 'straight_spread_left'; m_s = 'curled'; r_s = 'curled'; p_s = 'curled'
    elif label == 'R': t_s = 'across'; i_s = 'cross_right'; m_s = 'cross_left'; r_s = 'curled'; p_s = 'curled'
    elif label in ['S', 'yes']: t_s = 'across'; i_s = 'curled'; m_s = 'curled'; r_s = 'curled'; p_s = 'curled'
    elif label == 'T': t_s = 'tucked'; i_s = 'curled'; m_s = 'straight'; r_s = 'straight'; p_s = 'straight'
    elif label == 'U': t_s = 'across'; i_s = 'straight'; m_s = 'straight'; r_s = 'curled'; p_s = 'curled'
    elif label == 'V': t_s = 'across'; i_s = 'straight_spread_left'; m_s = 'straight_spread_right'; r_s = 'curled'; p_s = 'curled'
    elif label == 'W': t_s = 'across'; i_s = 'straight_spread_left'; m_s = 'straight'; r_s = 'straight_spread_right'; p_s = 'curled'
    elif label == 'X': t_s = 'across'; i_s = 'curved'; m_s = 'curled'; r_s = 'curled'; p_s = 'curled'
    elif label == 'Y': t_s = 'out'; i_s = 'curled'; m_s = 'curled'; r_s = 'curled'; p_s = 'straight'
    elif label == 'Z': t_s = 'across'; i_s = 'straight'; m_s = 'curled'; r_s = 'curled'; p_s = 'curled'
    elif label == 'Stop': t_s = 'out'; i_s = 'straight_spread_left'; m_s = 'straight'; r_s = 'straight_spread_right'; p_s = 'straight_spread_right'
    elif label == 'help': t_s = 'up'; i_s = 'curled'; m_s = 'curled'; r_s = 'curled'; p_s = 'curled'
    elif label == 'i love you': t_s = 'out'; i_s = 'straight'; m_s = 'curled'; r_s = 'curled'; p_s = 'straight'
    elif label == 'no': t_s = 'touch_index'; i_s = 'straight'; m_s = 'straight'; r_s = 'curled'; p_s = 'curled'
    else: t_s = 'across'; i_s = 'straight'; m_s = 'straight'; r_s = 'straight'; p_s = 'straight'

    set_thumb(t_s)
    set_index(i_s)
    set_middle(m_s)
    set_ring(r_s)
    set_pinky(p_s)
            
    return landmarks

def apply_augmentations(landmarks):
    scale = np.random.uniform(0.7, 1.3)
    aug_lm = landmarks * scale
    
    theta = np.deg2rad(np.random.uniform(-30, 30))
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))
    aug_lm = np.dot(aug_lm, rotation_matrix.T)
    
    shift_x = np.random.uniform(0.3, 0.7)
    shift_y = np.random.uniform(0.5, 0.9)
    aug_lm = aug_lm / 500.0 
    aug_lm[:, 0] += shift_x
    aug_lm[:, 1] += shift_y
    
    noise = np.random.normal(0, 0.005, size=aug_lm.shape)
    aug_lm += noise
    return aug_lm

def generate_dataset():
    output_path = "dataset.csv"
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)
        
        for label in LABELS:
            base_temp = get_base_template(label)
            for _ in range(SAMPLES_PER_CLASS):
                aug_lm = apply_augmentations(base_temp)
                
                row = [label]
                for i in range(21):
                    z = np.random.normal(0, 0.01)
                    row.extend([aug_lm[i, 0], aug_lm[i, 1], z])
                writer.writerow(row)
                
    print(f"Dataset generated successfully at {output_path} with {len(LABELS) * SAMPLES_PER_CLASS} samples.")

if __name__ == "__main__":
    generate_dataset()
