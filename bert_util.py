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