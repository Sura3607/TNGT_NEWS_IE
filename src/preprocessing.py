import pandas as pd
import re
import json
import os

class DataPipeline:
    def __init__(self, input_path: str, output_path: str, save_table: bool = True, window_size: int = 3, step_size: int = 2):
        """
        input_path: Đường dẫn file CSV đầu vào.
        output_path: Đường dẫn file JSON đầu ra.
        save_table: Có lưu bảng CSV các cửa sổ đã tách không (mặc định True).
        window_size: Số lượng câu trong 1 task (mặc định 3).
        step_size: Bước nhảy (mặc định 2).
        """
        self.input_path = input_path
        self.output_path = output_path
        self.save_table = save_table
        self.window_size = window_size
        self.step_size = step_size
        
        self.prep_dir = "data/preprocessed/"
        os.makedirs(self.prep_dir, exist_ok=True)

        self.abbreviations = {
            "TP.": "TP<PRD>", 
            "Tp.": "Tp<PRD>",
            "Mr.": "Mr<PRD>", 
            "Mrs.": "Mrs<PRD>",
            "Dr.": "Dr<PRD>", 
            "Th.S": "Th.S<PRD>", 
            "TS.": "TS<PRD>"
        }

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.replace('\n', ' ').replace('\r', ' ')

        #loại bỏ 1 vài credit phổ biến nhưng không hết hẳn
        upper = "A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ"
        lower = "a-zàáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵýỷỹ"
        pattern_credit = rf'(Video|Ảnh)\s*:\s*((?:[{upper}][{lower}]*\s+){{1,4}})(?=[{upper}])'
        text = re.sub(pattern_credit, '', text)
        text = re.sub(r'(Video|Ảnh)\s*:\s*', '', text)

        return re.sub(r'\s+', ' ', text).strip()

    def protect_abbreviations(self, text):
        for k, v in self.abbreviations.items():
            text = text.replace(k, v)
        return text

    def restore_abbreviations(self, text):
        return text.replace("<PRD>", ".")

    def split_sentences(self, text):
        if not text:
            return []
        
        text = self.protect_abbreviations(text)

        pattern = r'(?<=[.?!])\s+(?=[A-ZÀ-Ỹ])'
        sentences = re.split(pattern, text)

        results = []
        for s in sentences:
            s = self.restore_abbreviations(s).strip()
            if len(s) > 10:
                results.append(s)
        return results

    def run(self):
        print(f"Đang xử lý file: {self.input_path} (Window={self.window_size}, Step={self.step_size})...")

        df = pd.read_csv(self.input_path)
        df.columns = [c.upper() for c in df.columns]

        df['clean_title'] = df['TITLE'].apply(self.clean_text)
        df['clean_content'] = df['CONTENT'].apply(self.clean_text)

        df['full_text'] = df['clean_title'] + ". " + df['clean_content']

        cleaned_path = os.path.join(
            self.prep_dir,
            os.path.basename(self.input_path).replace(".csv", "_cleaned.csv")
        )

        if self.save_table:
            if 'SOURCE' in df.columns:
                df_save = df[['ID', 'full_text', 'SOURCE']].copy()
            else:
                df_save = df[['ID', 'full_text']].copy()
                df_save['SOURCE'] = ''

            df_save.columns = ['id', 'text', 'source']
            df_save.to_csv(cleaned_path, index=False, encoding='utf-8')
            print(f"-> Đã lưu CLEANED tại: {cleaned_path}")

        window_rows = [] # Dùng để lưu file CSV bảng (Mirror của JSON)
        tasks = []      # Dùng để lưu file JSON Label Studio

        for idx, row in df.iterrows():
            article_id = str(row['ID'])
            # Tách thành danh sách câu đơn lẻ trước
            sentences = self.split_sentences(row['full_text'])

            # bắt đầu trượt 
            window_idx = 0
            for i in range(0, len(sentences), self.step_size):
                # Lấy ra cửa sổ gồm 'window_size' câu
                chunk_sents = sentences[i : i + self.window_size]
                
                if not chunk_sents: 
                    continue

                chunk_text = " ".join(chunk_sents)
                
                chunk_id = f"{article_id}_w{window_idx}"

                # Thêm vào JSON List
                tasks.append({
                    "data": {
                        "text": chunk_text,
                        "ref_id": chunk_id,
                        "article_id": article_id
                    }
                })

                # Thêm vào CSV Table List (Đồng bộ với JSON)
                window_rows.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "article_id": article_id
                })
                window_idx += 1
        
        # Lưu file CSV dạng bảng của các Windows (để kiểm tra dễ hơn đọc JSON)
        split_path = os.path.join(
            self.prep_dir,
            os.path.basename(self.input_path).replace(".csv", "_cleaned_split.csv")
        )
        pd.DataFrame(window_rows).to_csv(split_path, index=False, encoding='utf-8')
        print(f"-> Đã lưu CLEANED_SPLIT (Bảng các Windows) tại: {split_path}")

        # Lưu file JSON import Label Studio
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        print(f"\nHoàn tất xử lý.")
        print(f"Tổng số bài viết: {len(df)}")
        print(f"Tổng số Task (Windows) tạo ra: {len(tasks)}")
        print(f"File JSON Import: {self.output_path}")

 
if __name__ == "__main__":
    pipeline = DataPipeline(
        input_path='data/raw/data_raw_400news.csv', 
        output_path='data/label_studio/400news_import.json',
        save_table=True,
        window_size=3, # Gộp 3 câu
        step_size=2    # Nhảy 2 câu mỗi lần
    )
    pipeline.run()