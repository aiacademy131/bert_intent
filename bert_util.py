from tqdm import tqdm
import numpy as np

def load_excel_data(work_book, sheet_name, menu_dict, temp_dict):
  if sheet_name not in menu_dict.values():
    menu_dict[str(len(menu_dict))] = sheet_name
  
  train = []
  sheet = work_book[sheet_name]
  for column_index, column in enumerate(sheet.columns):
    for index, row in enumerate(column):
      if index is 0:
        temp_dict[str(column_index)] = row.value
      else:
        if row.value:
          data = (len(menu_dict)-1, column_index, row.value)
          train.append(data)
  return train


def convert_data(data_df):
    global tokenizer
    global SEQ_LEN
    global LABEL_COLUMN
    global DATA_COLUMN
    
    tokens, masks, segments, targets = [], [], [], []
    
    for i in tqdm(range(len(data_df)), position=0, leave=True):
        # token : 문장을 토큰화함
        token = tokenizer.encode(data_df[DATA_COLUMN][i], max_length=SEQ_LEN, truncation = True, padding='max_length')
       
        # 마스크는 토큰화한 문장에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 통일
        num_zeros = token.count(0)
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
        
        # 문장의 전후관계를 구분해주는 세그먼트는 문장이 1개밖에 없으므로 모두 0
        segment = [0]*SEQ_LEN

        # 버트 인풋으로 들어가는 token, mask, segment를 tokens, segments에 각각 저장
        tokens.append(token)
        masks.append(mask)
        segments.append(segment)
        
        # 정답(긍정 : 1 부정 0)을 targets 변수에 저장해 줌
        targets.append(data_df[LABEL_COLUMN][i])

    # tokens, masks, segments, 정답 변수 targets를 numpy array로 지정    
    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    targets = np.array(targets)

    return [tokens, masks, segments], targets

# 위에 정의한 convert_data 함수를 불러오는 함수를 정의
def load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_df[LABEL_COLUMN] = data_df[LABEL_COLUMN].astype(int)
    data_x, data_y = convert_data(data_df)
    return data_x, data_y