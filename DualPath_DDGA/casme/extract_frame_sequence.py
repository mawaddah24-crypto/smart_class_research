import os
import shutil
import pandas as pd

def extract_frame_sequence(cropped_dir, excel_path, output_dir):
    df = pd.read_excel(excel_path)

    os.makedirs(output_dir, exist_ok=True)

    for idx, row in df.iterrows():
        subject = row['Subject']
        filename = row['Filename']
        onset = int(row['OnsetFrame'])
        apex = int(row['ApexFrame'])
        offset = int(row['OffsetFrame'])
        emotion = str(row['Estimated Emotion']).lower()

        src_folder = os.path.join(cropped_dir, subject, filename)
        if not os.path.exists(src_folder):
            print(f"⚠️ Folder tidak ditemukan: {src_folder}")
            continue

        # Create dest folder
        dst_seq_dir = os.path.join(output_dir, emotion, f"{subject}_{filename}")
        os.makedirs(dst_seq_dir, exist_ok=True)

        # Copy frames onset → offset
        for frame_num in range(onset, offset + 1):
            matched_file = None
            for file in os.listdir(src_folder):
                digits = ''.join(filter(str.isdigit, file))
                if digits and int(digits) == frame_num:
                    matched_file = file
                    break

            if matched_file:
                src = os.path.join(src_folder, matched_file)
                dst = os.path.join(dst_seq_dir, matched_file)
                shutil.copyfile(src, dst)
            else:
                print(f"❌ Frame {frame_num} tidak ditemukan di {src_folder}")

    print("\n🎉 Extraction selesai!")

if __name__ == "__main__":
    cropped = "../Cropped"
    coding_excel = "../CASME2-coding-20140508.xlsx"
    output = "../CASME2_Sequences"
    extract_frame_sequence(cropped, coding_excel, output)
