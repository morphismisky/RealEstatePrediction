# 必要なライブラリをインポート
import numpy as np  # 数値計算のためのNumPyライブラリ
import polars as pl  # 高速なデータ処理のためのPolarsライブラリ
import pandas as pd  # データ分析のためのPandasライブラリ
import re  # 正規表現を使用するためのライブラリ


def rename_columns(df: pl.DataFrame):
   """
   日本語の列名を英語に変換する関数
   
   入力:
       df (pl.DataFrame): 日本語の列名を持つPolarsデータフレーム
   
   処理:
       あらかじめ定義された辞書を使用して、日本語の列名を対応する英語名に変換
       
   出力:
       pl.DataFrame: 列名が英語に変換されたPolarsデータフレーム
   """
   rename_dict = {
       '賃料': 'Target',
       '所在地': 'Location',
       'アクセス': 'Access',
       '間取り': 'Floor_Plan',
       '築年数': 'Age_of_Building',
       '方角': 'Direction',
       '面積': 'Area',
       '所在階': 'Story_and_Floor',
       'バス・トイレ': 'Bath_and_Toilet',
       'キッチン': 'Kitchen',
       '放送・通信': 'Broadcasting_and_Communication',
       '室内設備': 'Indoor_Facilities',
       '駐車場': 'Parking',
       '周辺環境': 'Surrounding_Environment',
       '建物構造': 'Architecture',
       '契約期間': 'Contract_Period'
   }
   df = df.rename(rename_dict)
   return df

def process_Location(df: pl.DataFrame, geo_path='/Users/aria/Kaggle/real_estate/dataset/geoencoded_districts.csv', land_path='/Users/aria/Kaggle/real_estate/dataset/Jika_2019.csv'):
    """
    住所データを処理し、地理情報と地価情報を追加する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
        geo_path (str): 地理情報を含むCSVファイルのパス 
                       (出典: 国土交通省 国土数値情報 https://nlftp.mlit.go.jp/)
        land_path (str): 地価情報を含むCSVファイルのパス
                        (出典: 東京都財務局 地価公示 https://www.zaimu.metro.tokyo.lg.jp/kijunchi/chikakouji/31kouji)
    
    処理:
        1. 区の情報のみを残す
        2. 各住所の緯度経度情報の挿入
        3. 地価情報の挿入
    
    出力:
        pl.DataFrame: 地理情報と地価情報が追加されたデータフレーム
    """

    # CONST
    col_name = 'Location'
    # 漢数字からアラビア数字へのマッピング
    kanji_to_arabic = {
        '一': '1',
        '二': '2',
        '三': '3',
        '四': '4',
        '五': '5',
        '六': '6',
        '七': '7',
        '八': '8',
        '九': '9',
        '十': '10'  # 簡略化のため10までとする
    }

    # 漢数字をアラビア数字に置換する関数
    def replace_kanji_numerals(text):
        # 「丁目」の前の漢数字を検索する正規表現
        pattern = re.compile(r'([一二三四五六七八九十])丁目')
        matches = pattern.finditer(text)
        for match in matches:
            kanji_num = match.group(1)  # 漢数字
            arabic_num = kanji_to_arabic[kanji_num]  # 対応するアラビア数字
            text = text.replace(kanji_num + '丁目', arabic_num + '丁目')  # 元の文字列内を置換

        # 「番町」の前の漢数字を検索する正規表現
        pattern = re.compile(r'([一二三四五六七八九十])番町')
        matches = pattern.finditer(text)
        for match in matches:
            kanji_num = match.group(1)  # 漢数字
            arabic_num = kanji_to_arabic[kanji_num]  # 対応するアラビア数字
            text = text.replace(kanji_num + '番町', arabic_num + '番町')  # 元の文字列内を置換
        return text

    # 建物番号やその他の詳細情報を削除する関数
    def remove_building_numbers(address):
        # 正規表現を使用して建物番号部分とその他の詳細を削除し、「丁目」の前の番号は保持
        return re.sub(r'(\d+-\d+.*|(\d+)(?!丁目).*)$', '', address).strip()

    # 地価データの住所部分を抽出する関数
    def Jika_extract_address_part(address):
        match = re.match(r"^(.*?)(\d+)", address)
        if match:
            return match.group(1).strip()
        return address

    # 欠損値を埋める関数
    def fill_missing_values(df, column, mapping_dict, key_col):
        # 辞書アイテムをタプルのリストに変換
        mapping_list = list(mapping_dict.items())
    
        # 平均値用の一時的なDataFrameを作成
        mean_df = pl.DataFrame(mapping_list, schema=[key_col, 'mean_value'])
    
        # 一時的なDataFrameを元のDataFrameに結合
        df = df.join(mean_df, on=key_col, how='left')
    
        # 指定された列の欠損値を対応する平均値で埋める
        df = df.with_columns(
            pl.when(pl.col(column).is_null())
            .then(pl.col('mean_value'))
            .otherwise(pl.col(column))
            .alias(column)
        )
        # 一時的な'mean_value'列を削除
        df = df.drop('mean_value')
        return df
        
    # 住所データのクリーニング
    # 以下のデータクリーニング処理はトップソリューションを参考に実装
    # 参考: https://github.com/analokmaus/signate-studentcup2019/blob/master/preprocess/preprocess.py
    df = df.with_columns([
        pl.when(pl.col(col_name).str.contains(r'[\(（].*[）\)]'))
        .then(pl.col(col_name).str.replace(r'[\(（].*[）\)]', ''))
        .when(pl.col(col_name).str.contains(r'[\s、]'))
        .then(pl.col(col_name).str.replace(r'[\s、]', ''))
        .when(pl.col(col_name).str.contains('I'))
        .then(pl.col(col_name).str.replace('I', '1'))
        .otherwise(pl.col(col_name)).alias(col_name)
    ])
    ## 全角数字を半角数字に変換
    for zenkaku, hankaku in zip('０１２３４５６７８９', '0123456789'):
        df = df.with_columns(
            pl.col(col_name).str.replace(zenkaku, hankaku)
        )
    
    ## ハイフンの標準化と不要な文字の削除
    df = df.with_columns([
        pl.col(col_name)
            .str.replace('ー', '-')
            .str.replace(r'(以下|詳細)*未定', '', literal=True)
            .str.replace(r'[ー‐―−]', '-', literal=True)
            .str.replace(r'(\d)番[地]*', r'\1-', literal=True)
            .str.replace(r'[-]{2,}', r'-', literal=True)
            .str.replace(r'([a-zA-Z一-龥ぁ-んァ-ヶ・ー])(\d)', r'\1-\2', literal=True)
            .str.replace(r'(\d)号', r'\1', literal=True)
            .alias(col_name)
    ])

    # 番地を削除
    df = df.with_columns(
        pl.col(col_name).map_elements(remove_building_numbers)
    )

    # 区の情報を抽出
    df = df.with_columns(
        pl.col(col_name).str.extract("東京都(.+?)区").alias("district"),
        pl.when(pl.col(col_name).str.contains("丁目")).then(pl.col(col_name).str.extract(r"東京都(.+?)\d丁目"))
        .otherwise(pl.col(col_name).str.replace("東京都", ""))
        .alias("detailed_district"),
        pl.col(col_name).str.replace("東京都", "").alias(col_name)
    )

    # 都心5区かどうかのフラグを追加
    toshin_5districts = ['千代田', '中央', '港', '渋谷', '新宿']
    df = df.with_columns(
        pl.when(pl.col('district').is_in(toshin_5districts)).then(1).otherwise(0).alias('is_toshin')
    )

    # 緯度経度情報の取得と結合
    geoencoded_districts = pl.read_csv(geo_path, encoding='cp932')
    geoencoded_districts = geoencoded_districts.with_columns(
        (pl.col('市区町村名') + pl.col('大字町丁目名')).alias('district')
    ).select(
        'district',
        pl.col('緯度').alias('latitude'),
        pl.col('経度').alias('longitude')
    ).with_columns(
        pl.col('district').map_elements(replace_kanji_numerals)
    )

    # 緯度経度情報をデータフレームに結合
    df = df.join(
        geoencoded_districts, left_on=col_name, right_on='district', how='left'
    )

    # 地価データの準備（Pandas）
    jika = pd.read_csv(land_path, header=None, encoding="cp932")
    # 2行目（インデックス1）を新しいカラム名にする
    jika.columns = jika.iloc[1]  # 2行目をカラム名に設定
    jika = jika[2:].reset_index(drop=True)

    # 区の情報を抽出
    jika['区市町村名'] = jika['区市町村名'].fillna("")
    jika['地番'] = jika['地番'].fillna("")
    jika['district'] = (jika['区市町村名'] + jika['地番']).apply(Jika_extract_address_part)
    jika = pl.from_pandas(jika)

    # 地価データの前処理
    jika = jika.with_columns(
        pl.col('district').map_elements(replace_kanji_numerals),
        pl.col("当年価格（円）")
        .str.strip_chars()  # 前後の空白を除去
        .str.replace_all(",", "")  # カンマを削除
        .cast(pl.Float64, strict=False)  # 型変換（strict=False でエラー回避）
        .alias("Land_Price")
    )
    jika = jika[['district', "Land_Price"]]

    # 地価情報をデータフレームに結合
    df = df.join(
        jika, left_on=col_name, right_on='district', how='left'
    )

    # 欠損値を埋める（近隣の平均地価）
    grouped_df = df.group_by('detailed_district').agg(
        pl.col('Land_Price').mean().alias('mean_Land_Price')
    )
    mean_jika_dict = grouped_df.to_dict(as_series=False)
    mean_jika_dict = dict(zip(mean_jika_dict['detailed_district'], mean_jika_dict['mean_Land_Price']))
    df = fill_missing_values(df, 'Land_Price', mean_jika_dict, 'detailed_district')

    # 欠損値を埋める（区の平均地価で埋める）
    grouped_df = df.group_by('district').agg(
        pl.col('Land_Price').mean().alias('mean_Land_Price')
    )
    mean_jika_dict = grouped_df.to_dict(as_series=False)
    mean_jika_dict = dict(zip(mean_jika_dict['district'], mean_jika_dict['mean_Land_Price']))
    df = fill_missing_values(df, 'Land_Price', mean_jika_dict, 'district')

    # 不要な列を削除
    df = df.drop('detailed_district')
    
    return df

def process_Access(df: pl.DataFrame):
    """
    駅の情報を抽出し、駅へのアクセス性に関する特徴量を生成する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        1. 路線名を統一
        2. 最寄り駅への徒歩時間を抽出
        3. 物件の最寄り路線名を特定
        4. 駅近評価指標を作成（5分以内:2、10分以内:1、それ以外:0）
    
    出力:
        pl.DataFrame: 駅アクセス情報が処理されたデータフレーム
    """

    # 全角数字を半角数字に変換
    for zenkaku, hankaku in zip('０１２３４５６７８９', '0123456789'):
        df = df.with_columns(
            pl.col("Access").str.replace(zenkaku, hankaku)
        )

    # 一位の手法から引用
    ## リンク：https://github.com/analokmaus/signate-studentcup2019/blob/master/preprocess/preprocess.py
    ## 路線名の統一
    df = df.with_columns(
        pl.when(pl.col("Access").str.contains("湘南新宿ライン"))
            .then(pl.lit("湘南新宿ライン"))
        .when(pl.col("Access").str.contains("中央本線"))
            .then(pl.lit("中央本線"))
        .when(pl.col("Access") == "東武伊勢崎大師線")
            .then(pl.lit("東武スカイツリーライン"))
        .when(pl.col("Access") == "東武伊勢崎線")
            .then(pl.lit("東武伊勢崎線(押上－曳舟)"))
        .when(pl.col("Access") == "京王井の頭線")
            .then(pl.lit("井ノ頭線"))
        .when(pl.col("Access") == "京成電鉄本線")
            .then(pl.lit("京成本線"))
        .when(pl.col("Access") == "三田線")
            .then(pl.lit("都営三田線"))
        .when(pl.col("Access").is_in(["京王小田急線", "小田急電鉄小田原線"]))
            .then(pl.lit("小田急小田原線"))
        .when(pl.col("Access").is_in(["中央総武線", "中央総武緩行線"]))
            .then(pl.lit("総武線・中央線（各停）"))
        .when(pl.col("Access").is_in(["地下鉄浅草線", "浅草線"]))
            .then(pl.lit("都営浅草線"))
        .when(pl.col("Access") == "西武池袋豊島線")
            .then(pl.lit("西武池袋線"))
        .when(pl.col("Access") == "総武線")
            .then(pl.lit("総武線・中央線（各停）"))
        .when(pl.col("Access") == "東京臨海高速鉄道")
            .then(pl.lit("りんかい線"))
        .when(pl.col("Access") == "東京モノレール")
            .then(pl.lit("東京モノレール羽田線"))
        .when(pl.col("Access") == "日暮里舎人ライナー")
            .then(pl.lit("日暮里・舎人ライナー"))
        .when(pl.col("Access") == "大井町線")
            .then(pl.lit("東急大井町線"))
        .when(pl.col("Access") == "千代田常磐緩行線")
            .then(pl.lit("常磐線"))
        .when(pl.col("Access") == "中央線")
            .then(pl.lit("中央線（快速）"))
        .when(pl.col("Access") == "丸ノ内線")
            .then(pl.lit("丸ノ内線(池袋－荻窪)"))
        .when(pl.col("Access") == "京成成田空港線")
            .then(pl.lit("京成本線"))
        .when(pl.col("Access") == "京浜東北根岸線")
            .then(pl.lit("京浜東北線"))
        .otherwise(pl.col("Access"))  # 条件に一致しない場合は元の値を保持
        .alias("Access")
    )

    # アクセス情報から駅までの時間と路線名を抽出(独自実装)
    df = df.join(
        df.with_columns(
                pl.col('Access').str.split(by='\t\t').alias('Accessable_station_list')
        ).explode('Accessable_station_list').select(
            'id',
            # 「徒歩X分」から数字部分を抽出
            pl.col('Accessable_station_list').str.extract(r'徒歩(\d+)分').cast(pl.UInt32).alias('time_to_alive'),
            # 路線名を抽出
            pl.col('Accessable_station_list').str.extract(r'(\S*線)').alias('line_name')
        ).sort(by='time_to_alive')  # 徒歩時間で並べ替え
        .group_by('id').agg(
            pl.first('time_to_alive').alias('min_time_to_alive'),  # 最も近い駅までの時間
            pl.first('line_name').alias('nearest_line_name'),     # 最寄りの路線名
            pl.col('line_name').n_unique().alias('num_of_lines')  # アクセス可能な路線数
        ).select(
            'id',                # 物件のID
            'min_time_to_alive', # 最寄りとの距離
            'nearest_line_name', # 最寄り路線の名前
            'num_of_lines'       # 物件の中で言及されている駅の数
        ).with_columns(
            # 駅近評価指標の生成
            pl.when((pl.col('min_time_to_alive') <= 5)).then(2)   # 5分以内なら2
            .when((pl.col('min_time_to_alive') <= 10)).then(1)    # 10分以内なら1
            .otherwise(0).alias('close_to_station_evaluation')    # それ以外は0
        )
    , on='id', how='left')

    ## 最寄り路線名の不要な文字列を削除・標準化（引用）
    void_list = ['東京メトロ', 'JR', 'ＪＲ', '東京都', '都電', '都営', '東急']
    for word in void_list:
        df = df.with_columns(
            pl.when(pl.col('nearest_line_name').str.contains(word))
              .then(pl.col('nearest_line_name').str.replace(word, ''))
              .otherwise(pl.col('nearest_line_name'))
              .alias('nearest_line_name')
        )
    
    ## 路線名の表記ゆれを修正（引用）
    df = df.with_columns(
            pl.when(pl.col('nearest_line_name').str.contains('丸の内'))
              .then(pl.col('nearest_line_name').str.replace('丸の内', '丸ノ内'))
              .when(pl.col('nearest_line_name').str.contains('総武本線'))
              .then(pl.col('nearest_line_name').str.replace('総武本線', '総武線'))
              .when(pl.col('nearest_line_name').str.contains("京王井ノ頭"))
              .then(pl.col('nearest_line_name').str.replace('京王井ノ頭', '井ノ頭'))
              .when(pl.col('nearest_line_name').str.contains('"井の頭"'))
              .then(pl.col('nearest_line_name').str.replace('井の頭', '井ノ頭'))
              .otherwise(pl.col('nearest_line_name'))
              .alias('nearest_line_name')
    )

    # 元のアクセス情報列を削除
    df = df.drop('Access')

    return df

def process_Floor_Plan(df: pl.DataFrame):
    """
    間取り情報から部屋構成に関する特徴量を抽出する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        間取り情報（Floor_Plan列）から以下の特徴量を抽出する:
        1. 居室数（Num_of_rest_Rooms）
        2. リビングの有無（Having_Living）
        3. ダイニングの有無（Having_Dining）
        4. キッチンの有無（Having_Kitchen）
        5. 物置の有無（Having_Store）
        6. ワンルームかどうか（Having_Room）
        7. 部屋の総数（Num_of_Rooms）
    
    出力:
        pl.DataFrame: 間取り情報から特徴量が抽出されたデータフレーム
    """
    # 間取り情報から各種特徴量を抽出
    df = df.with_columns(
        # 居室数を抽出（間取り表記の最初の数字）
        pl.col('Floor_Plan').apply(lambda x: int(x[0]) if x[0].isdigit() else None).alias('Num_of_rest_Rooms'),
        # リビングの有無 (Lを含む場合は1、そうでなければ0)
        pl.when(pl.col('Floor_Plan').str.contains('L')).then(1).otherwise(0).alias('Having_Living'),
        # ダイニングの有無 (Dを含む場合は1、そうでなければ0)
        pl.when(pl.col('Floor_Plan').str.contains('D')).then(1).otherwise(0).alias('Having_Dining'),
        # キッチンの有無 (Kを含む場合は1、そうでなければ0)
        pl.when(pl.col('Floor_Plan').str.contains('K')).then(1).otherwise(0).alias('Having_Kitchen'),
        # 物置の有無 (Sを含む場合は1、そうでなければ0)
        pl.when(pl.col('Floor_Plan').str.contains('S')).then(1).otherwise(0).alias('Having_Store'),
        # ワンルームかどうか (Rを含む場合は1、そうでなければ0)
        pl.when(pl.col('Floor_Plan').str.contains('R')).then(1).otherwise(0).alias('Having_Room'),
    ).with_columns(
        # 部屋の総数を計算
        # ワンルームの場合は1、それ以外は居室数+L+D+K+Sの合計
        pl.when(pl.col('Having_Room') > 0).then(1)
        .otherwise(pl.col('Num_of_rest_Rooms') + 
                  (pl.col('Having_Living') + 
                   pl.col('Having_Dining') + 
                   pl.col('Having_Kitchen') + 
                   pl.col('Having_Store'))).alias('Num_of_Rooms')
    )
    
    # 元の間取り情報列を削除
    df = df.drop('Floor_Plan')
    
    return df

def process_Age_of_Building(df: pl.DataFrame):
    """
    築年数情報を月数に変換し、築年数評価指標を作成する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        1. 築年数の「年」と「ヶ月」を抽出して合計月数に変換
           例: '3年2ヶ月' → 3*12 + 2 = 38ヶ月
        2. 新築の場合は0ヶ月として処理
        3. 築年数のランク付け（10年未満:2、20年未満:1、それ以上:0）
    
    出力:
        pl.DataFrame: 築年数が月数に変換され、評価指標が追加されたデータフレーム
    """
    # 年月を合計月数に変換する関数
    def calculate_months(s):
        # 年数を抽出（存在しない場合は0）
        years = s.str.extract(r'(\d+)\s*年', 1).fill_null(0).cast(pl.Int32)
        # 月数を抽出（存在しない場合は0）
        months = s.str.extract(r'(\d+)\s*ヶ月', 1).fill_null(0).cast(pl.Int32)
        # 合計月数を計算
        return years * 12 + months
    
    # 新築は0ヶ月、それ以外は関数を適用して合計月数を算出
    df = df.with_columns(
        pl.when(pl.col('Age_of_Building').str.contains('新築')).then(0)
        .otherwise(calculate_months(pl.col('Age_of_Building'))).alias('total_months')
    ).with_columns(
        # 築年数の評価指標を作成
        pl.when(pl.col('total_months') < 10*12).then(2)   # 10年未満は2
        .when(pl.col('total_months') < 20*12).then(1)     # 20年未満は1
        .otherwise(0).alias('recommended_AoB')            # それ以上は0
    )
    
    # 元の築年数列を削除
    df = df.drop('Age_of_Building')
    
    return df

def process_Area(df: pl.DataFrame):
    """
    面積情報から数値のみを抽出する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        'Area'列から数値部分のみを抽出し、浮動小数点数に変換
        例: '25.5m2' → 25.5
    
    出力:
        pl.DataFrame: 面積が数値化されたデータフレーム
    """
    # 面積から数値部分を抽出する関数
    def calculate_area(s):
        # 正規表現で数値部分を抽出し浮動小数点数に変換
        area = s.str.extract(r'(\d+(?:\.\d+)?)m2', 1).cast(pl.Float64)
        return area
    
    # 関数を適用して面積を数値化
    df = df.with_columns(
        calculate_area(pl.col('Area')).alias('Area')
    )
    
    return df

def process_Story_and_Floor(df: pl.DataFrame):
    """
    建物の階数情報を処理する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        1. 建物の最大階数を抽出
        2. 物件がある階数を抽出
        3. 地下階の有無を判定
        4. 階数比率と自室の階数比率を計算
    
    出力:
        pl.DataFrame: 階数情報が処理されたデータフレーム
    """
    # 建物の最大階数、物件所在階、地下階の有無を抽出
    df = df.with_columns(
        # 「X階建」から最大階数を抽出
        pl.when(pl.col('Story_and_Floor').str.contains('階建')).then(pl.col('Story_and_Floor').str.extract(r'(\d+)階建').cast(pl.UInt32)).otherwise(None).alias('max_floor'),
        # 「／」の前にある「X階」から所在階を抽出
        pl.when(pl.col('Story_and_Floor').str.contains('／')).then(pl.col('Story_and_Floor').str.extract(r'(\d+)階').cast(pl.UInt32)).otherwise(None).alias('own_floor'),
        # 「地下X階」から地下階の有無を判定（あれば階数、なければ0）
        pl.when(pl.col('Story_and_Floor').str.contains('地下')).then(pl.col('Story_and_Floor').str.extract(r'地下(\d+)階').cast(pl.UInt32)).otherwise(0).alias('Having_under_floor')
    )

    # 階数比率の計算
    df = df.with_columns(
        # 自室の階数比率（最大階数に対する割合）を計算
        pl.when((pl.col('own_floor').is_null()) & (pl.col('max_floor').is_null().not_())).then(1)
        .when((pl.col('own_floor').is_null().not_()) & (pl.col('max_floor').is_null().not_())).then(1/(pl.col('max_floor'))).otherwise(None).alias('own_rooms_ratio'),
        # 実際の階数比率（所在階/最大階数）
        (pl.col('own_floor')/pl.col('max_floor')).alias('floor_ratio')
    )

    # 元の階数情報列を削除
    df = df.drop('Story_and_Floor')
    
    return df

def process_Bath_and_Toilet(df: pl.DataFrame):
    """
    トイレと風呂の機能情報を処理する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        1. トイレと風呂の機能数をカウント
        2. 風呂・トイレが別かどうかを判定
    
    出力:
        pl.DataFrame: バス・トイレ情報が処理されたデータフレーム
    """
    # タブと区切り文字の置換
    df = df.with_columns(
        pl.col("Bath_and_Toilet").str.replace_all("\t", " ")
    )
    df = df.with_columns(
        pl.col("Bath_and_Toilet").str.replace_all("／","")
    )

    # トイレと風呂の機能リスト
    bath_functions = ['専用バス', 'シャワー', '追焚機能', '洗面台独立', '脱衣所', '浴室乾燥機']
    toilet_functions = ['専用トイレ', '温水洗浄便座']

    # 機能の一致数をカウントする関数
    def count_matches(lst, value_list):
        if lst is None:
            return 0
        num_of_func = sum(1 for x in lst if x in value_list)
        return num_of_func
        
    # 機能数のカウントと風呂・トイレの分離状況を判定
    df = df.with_columns(
        # トイレの機能数をカウント
        pl.col('Bath_and_Toilet')
        .str.split(' ')
        .apply(lambda lst: count_matches(lst, toilet_functions))
        .alias('toilet_functions'),

        # 風呂の機能数をカウント
        pl.col('Bath_and_Toilet')
        .str.split(' ')
        .apply(lambda lst: count_matches(lst, bath_functions))
        .alias('bath_functions'),

        # 風呂・トイレが別かどうかを判定
        pl.when(pl.col('Bath_and_Toilet').str.contains('別'))
        .then(1)
        .otherwise(0)
        .cast(pl.UInt32)
        .alias('is_separate')
    )

    # 欠損値を0で埋める
    df = df.with_columns(
        pl.col('toilet_functions').fill_null(strategy='zero').alias('toilet_functions'),
        pl.col('bath_functions').fill_null(strategy='zero').alias('bath_functions'),
        pl.col('is_separate').fill_null(strategy='zero').alias('is_separate')
    )

    # 元のバス・トイレ情報列を削除
    df = df.drop('Bath_and_Toilet')
    
    return df

def process_Broadcasting_and_Communication(df: pl.DataFrame):
    """
    通信環境に関する情報を処理する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        1. インターネット対応、光ファイバー、アンテナの有無を判定
        2. 放送・通信機能の総数をカウント
    
    出力:
        pl.DataFrame: 通信環境情報が処理されたデータフレーム
    """
    # 重要な通信サービスのフラグを作成
    df = df.with_columns(
        # インターネット対応の有無
        pl.when(pl.col('Broadcasting_and_Communication').str.contains('インターネット対応')).then(1).otherwise(0).alias('has_internet'),
        # 光ファイバーの有無
        pl.when(pl.col('Broadcasting_and_Communication').str.contains('光ファイバー')).then(1).otherwise(0).alias('has_fiber'),
        # アンテナの有無
        pl.when(pl.col('Broadcasting_and_Communication').str.contains('アンテナ')).then(1).otherwise(0).alias('has_antenna')
    )

    # タブと区切り文字の置換
    df = df.with_columns(
        pl.col("Broadcasting_and_Communication").str.replace_all("\t", " ")
    )
    df = df.with_columns(
        pl.col("Broadcasting_and_Communication").str.replace_all("／","")
    )
    
    # 通信機能の総数をカウント
    df = df.with_columns(
        pl.col('Broadcasting_and_Communication')
        .str.split(' ').map_elements(lambda x: len(x))
        .alias('num_of_BC_functions'),
    )

    # 元の通信情報列を削除
    df = df.drop('Broadcasting_and_Communication')
    
    return df

def process_Kitchen(df: pl.DataFrame):
    """
    キッチン情報を処理する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        1. ガスコンロ、IH、システムキッチンの有無を判定
        2. コンロの口数を抽出
        3. キッチンのランク付け（L字>カウンター>システム>その他）
        4. キッチン機能の数をカウント
    
    出力:
        pl.DataFrame: キッチン情報が処理されたデータフレーム
    """
    # 区切り文字の統一
    df = df.with_columns(
        pl.col("Kitchen").str.replace_all("／"," ")
    )

    # コンロの口数を抽出する関数
    def process_row(row):
        match = re.search(r'コンロ(\d+)口', row)
        if match:
            number = int(match.group(1))
            new_row = re.sub(r'コンロ\d+口', '', row).strip()
            return new_row, int(number)
        else:
            return row, None

    # キッチン情報を処理
    df = df.with_columns([
        # コンロの口数を抽出して、元の文字列からは削除
        pl.col('Kitchen').map_elements(lambda row: process_row(row)[0]).alias('Kitchen'),
        pl.col('Kitchen').map_elements(lambda row: process_row(row)[1]).alias('cock_number'),
        # キッチンのタイプによるランク付け
        pl.when(pl.col('Kitchen').str.contains('L字')).then(3)         # L字キッチン（最高ランク）
        .when(pl.col('Kitchen').str.contains('カウンター')).then(2)    # カウンターキッチン
        .when(pl.col('Kitchen').str.contains('システム')).then(1)      # システムキッチン
        .otherwise(0).alias('Kitchen_Ranking')                        # その他
    ])

    # キッチン機能の数をカウント
    df = df.with_columns(
        pl.col('Kitchen')
        .str.split(' ').map_elements(lambda x: len(x), return_dtype=pl.Int64)
        .alias('Kitchen_feature_number'),
    )

    # キッチン設備のフラグを作成
    df = df.with_columns(
        # ガスコンロの有無
        pl.when(pl.col('Kitchen').str.contains('ガスコンロ'))
            .then(1)
            .otherwise(0)
            .cast(pl.Int32)
            .alias('has_gas_stove'),

        # IHコンロの有無
        pl.when(pl.col('Kitchen').str.contains('IH'))
            .then(1)
            .otherwise(0)
            .cast(pl.Int32)
            .alias('has_IH_stove'),

        # システムキッチンの有無
        pl.when(pl.col('Kitchen').str.contains('システムキッチン'))
            .then(1)
            .otherwise(0)
            .cast(pl.Int32)
            .alias('has_system_kitchen')
    )

    # 元のキッチン情報列を削除
    df = df.drop('Kitchen')
    
    return df

def process_Indoor_Facilities(df: pl.DataFrame):
    """
    室内設備情報を処理する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        1. エアコン、エレベーター、ごみ置き場、洗濯機置場、バルコニーなどの
           様々な室内設備の有無を判定
        2. 設備の総数をカウント
    
    出力:
        pl.DataFrame: 室内設備情報が処理されたデータフレーム
    """
    # 検出する設備とその列名を定義
    features = {
        'エアコン': 'has_air_conditioner',
        'エレベーター': 'has_elevator',
        '敷地内ごみ': 'has_garbage_area',
        '室内洗濯機': 'has_laundry_space',
        'バルコニー': 'has_balcony',
        '都市ガス': 'has_city_gus',
        '防音室': 'has_soundproof_room',
        '井戸': 'has_well',
        'ルーフバルコニー': 'has_roof_balcony',
        'ガス暖房': 'has_gas_heating',
        '室内洗濯機置場': 'has_indoor_laundry_space',
        '汲み取り': 'has_septic_tank',
        'シューズボックス': 'has_shoe_box',
        '室外洗濯機置場': 'has_outdoor_laundry_space',
        'エアコン付': 'has_air_conditioner_incl',
        'バリアフリー': 'has_barrier_free',
        'プロパンガス': 'has_propane_gas',
        '浄化槽': 'has_septic_system',
        '床暖房': 'has_floor_heating',
        # 追加の設備があれば同様のパターンで追加
    }

    # 各設備のワンホット列を作成
    feature_expressions = [
        pl.when(pl.col('Indoor_Facilities').str.contains(feature))
          .then(1)
          .otherwise(0)
          .alias(alias)
        for feature, alias in features.items()
    ]

    # 設備列をデータフレームに追加
    df = df.with_columns(*feature_expressions)

    # 設備の総数をカウント
    total_equipment = sum(pl.col(alias) for alias in features.values())
    df = df.with_columns(total_equipment.alias('num_of_equipments'))

    # 元の設備情報列を削除
    df = df.drop('Indoor_Facilities')

    return df

def process_Parking(df: pl.DataFrame):
    """
    駐車場情報を処理する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        1. 駐車場情報を統一形式に変換
        2. 車、自転車、バイクそれぞれの駐車場状況をランク付け
           - 空き有り: 2
           - 近隣にあり: 1
           - なし: 0
    
    出力:
        pl.DataFrame: 駐車場情報が処理されたデータフレーム
    """
    # タブ文字をスペースに置換
    df = df.with_columns(
        pl.col("Parking").str.replace_all("\t", " ")
    )

    # 各種駐車場の状況をランク付け
    df = df.with_columns(
        # 車の駐車場状況
        pl.when(pl.col('Parking').str.contains('駐車場 空有')).then(2)   # 空きあり
        .when(pl.col('Parking').str.contains('駐車場 近隣')).then(1)     # 近隣にあり
        .otherwise(0)                                                    # なし
        .cast(pl.UInt32)
        .alias('has_car_Parking'),

        # 自転車置き場の状況
        pl.when(pl.col('Parking').str.contains('駐輪場 空有')).then(2)   # 空きあり
        .when(pl.col('Parking').str.contains('駐輪場 近隣')).then(1)     # 近隣にあり
        .otherwise(0)                                                    # なし
        .cast(pl.UInt32)
        .alias('has_bycycle_Parking'),

        # バイク置き場の状況
        pl.when(pl.col('Parking').str.contains('バイク置き場 空有')).then(2)  # 空きあり
        .when(pl.col('Parking').str.contains('バイク置き場 近隣')).then(1)    # 近隣にあり
        .otherwise(0)                                                         # なし
        .cast(pl.UInt32)
        .alias('has_bike_Parking')
    )
    
    # 元の駐車場情報列を削除
    df = df.drop('Parking')
    
    return df

def process_Surrounding_Environment(df: pl.DataFrame):
    """
    周辺環境情報を処理する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        1. 最寄りのスーパーマーケットまでの距離を抽出
        2. 最寄りのコンビニエンスストアまでの距離を抽出
        3. 物件周辺の施設数をカウント
    
    出力:
        pl.DataFrame: 周辺環境情報が処理されたデータフレーム
    """
    # スーパーマーケットまでの距離を抽出する関数
    def find_super_distance(stations_str):
        # タブ文字で分割し、空文字を除外
        place_and_distances = [s for s in stations_str.split('\t') if s.strip()]
        # スーパーを含む項目のみを抽出
        place_and_distances = [s for s in place_and_distances if 'スーパー' in s]
    
        # 距離情報を数値に変換する関数
        def extract_super_time(entries):
            distances = []
            for entry in entries:
                distances.append(int(entry.split(' ')[1].replace('m', '')))
            return np.min(distances)  # 最小距離を返す
        
        # スーパーの情報があれば最小距離を返し、なければNoneを返す
        if len(place_and_distances) > 0:
            return extract_super_time(place_and_distances)
        else:
            return None
            
    # コンビニまでの距離を抽出する関数
    def find_convinience_store_distance(stations_str):
        # タブ文字で分割し、空文字を除外
        place_and_distances = [s for s in stations_str.split('\t') if s.strip()]
        # コンビニを含む項目のみを抽出
        place_and_distances = [s for s in place_and_distances if 'コンビニ' in s]
    
        # 距離情報を数値に変換する関数
        def extract_super_time(entries):
            distances = []
            for entry in entries:
                distances.append(int(entry.split(' ')[1].replace('m', '')))
            return np.min(distances)  # 最小距離を返す

        # コンビニの情報があれば最小距離を返し、なければNoneを返す
        if len(place_and_distances) > 0:
            return extract_super_time(place_and_distances)
        else:
            return None

    # 距離情報を抽出
    df = df.with_columns(
        # スーパーマーケットまでの距離
        pl.col('Surrounding_Environment').map_elements(find_super_distance).alias('super_distance'),
        # コンビニエンスストアまでの距離
        pl.col('Surrounding_Environment').map_elements(find_convinience_store_distance).alias('cs_distance'),
    )

    # 周辺の施設数をカウント（【】の数をカウント）
    df = df.with_columns(
        pl.col('Surrounding_Environment')
        .str.count_match('【')  # 【】で囲まれた施設の数をカウント
        .alias('count_buildings')
    )

    # 元の周辺環境情報列を削除
    df = df.drop('Surrounding_Environment')

    return df

def process_Architecture(df: pl.DataFrame):
    """
    建物構造情報を処理する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        不動産サイトを参考に、各構造をランク付け
        （数値が小さいほど耐久性や防音性が優れている）
        1: SRC（鉄骨鉄筋コンクリート）
        2: RC（鉄筋コンクリート）
        3: HPC（プレキャストコンクリート）
        4: PC（プレキャストコンクリート）
        5: ALC（軽量気泡コンクリート）
        6: 鉄骨造
        7: 軽量鉄骨
        8: 木造
        9: ブロック
        10: その他
    
    出力:
        pl.DataFrame: 建物構造がランク付けされたデータフレーム
    """
    # 建物構造のランク付け
    df = df.with_columns(
        pl.when(pl.col('Architecture').str.contains('SRC')).then(1)       # 鉄骨鉄筋コンクリート（最高ランク）
        .when(pl.col('Architecture').str.contains('RC')).then(2)          # 鉄筋コンクリート
        .when(pl.col('Architecture').str.contains('HPC')).then(3)         # プレキャストコンクリート（高品質）
        .when(pl.col('Architecture').str.contains('PC')).then(4)          # プレキャストコンクリート
        .when(pl.col('Architecture').str.contains('ALC')).then(5)         # 軽量気泡コンクリート
        .when(pl.col('Architecture').str.contains('鉄骨造')).then(6)      # 鉄骨造
        .when(pl.col('Architecture').str.contains('軽量鉄骨')).then(7)    # 軽量鉄骨
        .when(pl.col('Architecture').str.contains('木造')).then(8)        # 木造
        .when(pl.col('Architecture').str.contains('ブロック')).then(9)    # ブロック
        .otherwise(10).alias('rank_of_material')                          # その他
    )
    
    # 元の建物構造情報列を削除
    df = df.drop('Architecture')
    
    return df

def process_Contract_Period(df: pl.DataFrame, base_year=2019, base_month=4):
    """
    契約期間情報を処理する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
        base_year (int): 基準年（デフォルト: 2019年）
        base_month (int): 基準月（デフォルト: 4月）
    
    処理:
        1. 定期借家契約かどうかを判定
        2. 契約期間を月数に変換
           - 例: '2年' → 24ヶ月
           - 例: '1年6ヶ月' → 18ヶ月
           - 例: '2021年3月' → 基準日からの月数
    
    出力:
        pl.DataFrame: 契約期間情報が処理されたデータフレーム
    """
    # 期間表記を月数に変換する関数
    def convert_to_months(duration):
        # 「X年Yヶ月」の形式
        year_month_match = re.match(r'(\d+)年(\d+)ヶ月', duration)
        # 「X年」の形式
        year_match = re.match(r'(\d+)年', duration)
        # 「Xヶ月」の形式
        month_match = re.match(r'(\d+)ヶ月', duration)
        # 「YYYY年MM月」の形式（終了日）
        full_date_match = re.match(r'(\d+)年(\d+)月', duration)
    
        if year_month_match:
            # 「X年Yヶ月」を月数に変換
            years = int(year_month_match.group(1))
            months = int(year_month_match.group(2))
            return years * 12 + months
        elif full_date_match:
            # 「YYYY年MM月」を基準日からの月数に変換
            years = int(full_date_match.group(1)) - base_year
            months = int(full_date_match.group(2)) - base_month
            total_months = years * 12 + months
            return total_months
        elif year_match:
            # 「X年」を月数に変換
            years = int(year_match.group(1))
            return years * 12
        elif month_match:
            # 「Xヶ月」をそのまま返す
            months = int(month_match.group(1))
            return months

        return None

    # 契約情報の処理
    df = df.with_columns(
        # 定期借家契約かどうかを判定
        pl.when(pl.col('Contract_Period').str.contains('定期借家'))
        .then(1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias('is_temporal'),
        
        # 契約期間を月数に変換
        pl.col('Contract_Period').map_elements(convert_to_months).alias('term'),
    )

    # 元の契約期間情報列を削除
    df = df.drop('Contract_Period')

    return df

def delete_outlier(df: pl.DataFrame):
    """
    異常値を持つ行を削除する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        1. 不可能な値を持つ行を削除（例: 築年数が100年を超える）
        2. 外れ値を持つ行を削除（例: 賃料が極端に高い物件）
    
    出力:
        pl.DataFrame: 異常値が削除されたデータフレーム
    """
    # 以下の条件に該当する行を残す
    # 1. テストデータ（is_train=0）はすべて残す
    # 2. 訓練データ（is_train=1）の場合:
    #    - 築年数が100年未満
    #    - 単位面積あたりの賃料が30,000円以下または欠損値
    df = df.filter(
        (pl.col('is_train') == 0) | (
            (pl.col('total_months') < 12 * 100) & (
                (pl.col('Unit_Target') <= 30000) | (pl.col('Unit_Target').is_null())
            )
        )
    )

    return df

def modify_miss(df: pl.DataFrame):
    """
    誤った値や誤字脱字を修正する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        1. 既知の誤りを修正（参考: トップソリューション）
        2. 列ごとの一般的な誤りを修正
    
    出力:
        pl.DataFrame: 誤りが修正されたデータフレーム
    """
    # トップソリューションを参考にした修正リスト
    # https://github.com/analokmaus/signate-studentcup2019/blob/master/preprocess/preprocess.py
    train_fix = [
        [3335, 'Location', '東京都中央区晴海２丁目２－２－４２', '東京都中央区晴海２丁目２－４２'],
        [5776, 'Target', 1203500, 123500],
        [7089, 'Location', '東京都大田区池上８丁目8-6-2', '東京都大田区池上８丁目6-2'],
        [7492, 'Area', '5.83m2', '58.3m2'],
        [9483, 'Location', '東京都世田谷区太子堂一丁目', '東京都世田谷区太子堂1丁目'],
        [19366, 'Location', '東京都大田区池上８丁目8-6-2', '東京都大田区池上８丁目6-2'],
        [20232, 'Age_of_Building', '520年5ヶ月', '20年5ヶ月'],
        [20428, 'Age_of_Building', '1019年7ヶ月', '19年7ヶ月'],
        [20888, 'Location', '東京都大田区本羽田一丁目', '東京都大田区本羽田1丁目'],
        [20927, 'Area', '430.1m2', '43.1m2'],
        [21286, 'Location', '東京都北区西ケ原３丁目西ヶ原３丁目', '東京都北区西ケ原３丁目'],
        [22250, 'Location', '東京都中央区晴海２丁目２－２－４２', '東京都中央区晴海２丁目２－４２'],
        [22884, 'Location', '東京都新宿区下落合２丁目2-1-17', '東京都新宿区下落合２丁目1-17'],
        [27199, 'Location', '東京都中央区晴海２丁目２－２－４２', '東京都中央区晴海２丁目２－４２'],
        [28141, 'Location', '東京都北区西ケ原１丁目西ヶ原１丁目', '東京都北区西ケ原１丁目']
    ]
    test_fix = [
        [34519, 'Location', '東京都足立区梅田１丁目1-8-16', '東京都足立区梅田１丁目8-16'],
        [34625, 'Location', '東京都渋谷区千駄ヶ谷３丁目3-41-12', '東京都渋谷区千駄ヶ谷３丁目41-12'],
        [36275, 'Location', '東京都大田区本羽田一丁目', '東京都大田区本羽田1丁目'],
        [40439, 'Location', '東京都品川区東品川四丁目', '東京都品川区東品川4丁目'],
        [41913, 'Location', '東京都板橋区志村１丁目１－８－１', '東京都板橋区志村１丁目８－１'],
        [45863, 'Location', '東京都大田区東糀谷３丁目3-2-2', '東京都大田区東糀谷３丁目2-2'],
        [49887, 'Location', '東京都大田区大森北一丁目', '東京都大田区大森北1丁目'],
        [56202, 'Location', '東京都大田区新蒲田３丁目9--20', '東京都大田区新蒲田３丁目9-20'],
        [57445, 'Location', '東京都目黒区八雲二丁目', '東京都目黒区八雲2丁目'],
        [58136, 'Location', '東京都文京区本駒込６丁目１－２２－４０３', '東京都文京区本駒込６丁目１－２２'],
        [58987, 'Location', '東京都北区西ケ原４丁目西ヶ原４丁目', '東京都北区西ケ原４丁目']
    ]

    # IDに基づく固有の修正
    for id, col, _, new in train_fix + test_fix:
        df = df.with_columns(
            pl.when(pl.col('id') == id)
                .then(pl.lit(new))
                .otherwise(pl.col(col))
                .alias(col)
        )
    
    # 一般的な誤りの修正
    df = df.with_columns(
        # 間取りの修正
        pl.when(pl.col("Floor_Plan").str.contains("11R")).then(pl.lit("1R")).otherwise(pl.col("Floor_Plan")).alias("Floor_Plan"),
        # 築年数の修正
        pl.when(pl.col("Age_of_Building").str.contains("520年5ヶ月")).then(pl.lit("52年5ヶ月"))
            .when(pl.col("Age_of_Building").str.contains("1019年7ヶ月")).then(pl.lit("19年7ヶ月"))
            .otherwise(pl.col("Age_of_Building")).alias("Age_of_Building"),
        # 面積の修正
        pl.when(pl.col("Area") == "430.1m2").then(pl.lit("43.01m2"))
            .when(pl.col("Area") == "1m2").then(pl.lit("10m2"))
            .when(pl.col("Area") == "5.83m2").then(pl.lit("58.3m2"))
            .otherwise(pl.col("Area")).alias("Area"),
    )

    # 賃料の特定修正
    df = df.with_columns(
        pl.when(pl.col("id") == 5776).then(pl.lit(120350)).otherwise(pl.col("Target")).alias("Target")
    )

    return df

def fill_in_target(df: pl.DataFrame):
    """
    テストデータの目的変数を埋める関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        1. テストデータの目的変数を、同じ住所・階数・面積・地下階有無を持つ
           訓練データの目的変数で補完
        2. 複数の候補がある場合は、最大値と最小値の比率が1.1以下なら平均値を使用
    
    参考:
        2位のソリューション
    
    出力:
        pl.DataFrame: 目的変数が補完されたデータフレーム
    """
    # 訓練データとテストデータを分割
    df_train = df.filter(pl.col('is_train') == 1)
    df_test = df.filter(pl.col('is_train') == 0)
    
    # テストデータの目的変数列を削除
    df_test = df_test.drop('Target')
    
    # 訓練データから同じ条件の物件の目的変数を取得
    df_test = df_test.join(
        df_train.select(
            'Location',            # 住所
            'Area',                # 面積
            'max_floor',           # 最大階数
            'Having_under_floor',  # 地下階の有無
            'Target'               # 目的変数（賃料）
        ),
        on=['Location',
            'max_floor',
            'Area',
            'Having_under_floor',
            # 'total_months'       # 築年数（コメントアウト）
            ],
        how='left'
    )
    
    # 複数の候補がある場合の処理
    df_test = df_test.group_by('id').agg(
            # ID以外とTargetを除く全ての列を最初の値で集約
            [pl.first(col).alias(col) for col in df_test.drop(['id', 'Target']).columns] + \
            # 目的変数は最大値と最小値の比率が1.1以下なら平均値、それ以外はNull
            [pl.when(pl.max('Target')/pl.min('Target') <= 1.1).then(pl.mean('Target')).otherwise(None).alias('Target')]
        )
    
    # 訓練データの目的変数を浮動小数点に変換
    df_train = df_train.with_columns(
        pl.col('Target').cast(pl.Float64).alias('Target')
    )

    # テストデータと訓練データの列を揃える
    df_test = df_test.select(df_train.columns)
    
    # 訓練データとテストデータを結合
    df = pl.concat([df_train, df_test])

    return df

def ensemble_columns(df: pl.DataFrame):
    """
    複数の特徴量を組み合わせて新しい特徴量を生成する関数
    
    入力:
        df (pl.DataFrame): 処理対象のデータフレーム
    
    処理:
        1. ヴィンテージ物件フラグの作成
        2. 築年数の上限調整
        3. 単位面積あたりの賃料の計算
        4. 港区と中央区からの距離ポテンシャルの計算
    
    出力:
        pl.DataFrame: 新しい特徴量が追加されたデータフレーム
    """
    # 新しい特徴量の生成
    df = df.with_columns(
        # ヴィンテージ物件かどうか（広くて古くて設備が充実）
        pl.when((pl.col('Area') > 100) & (pl.col('total_months') > 40) & (pl.col('num_of_equipments') > 4)).then(1).otherwise(0).alias('is_vintage'),
        # 築年数の上限調整（面積が100平米超の新築っぽい物件の築年数は最低15ヶ月とする）
        pl.when((pl.col('Area') > 100) & (pl.col('total_months') < 15)).then(15).otherwise(pl.col('total_months')).alias('total_months'),
        # 単位面積あたりの賃料
        (pl.col('Target') / pl.col('Area')).alias('Unit_Target')
    )

    # 港区の平均緯度経度を取得
    Minato_lat, Minato_lon = df.filter(
        pl.col('district') == '港'
    ).select(
        pl.col('latitude').mean().alias('mean_lat'),
        pl.col('longitude').mean().alias('mean_lon')
    ).to_numpy().flatten()  # 1次元配列に変換
    
    # 港区からのポテンシャル（距離の逆二乗に比例）を計算
    df = df.with_columns(
        (1 / ((pl.col('latitude') - Minato_lat)**2 + (pl.col('longitude') - Minato_lon)**2)).alias('Minato_ward_Potential')
    )

    # 中央区の平均緯度経度を取得
    Chuou_lat, Chuou_lon = df.filter(
        pl.col('district') == '中央'
    ).select(
        pl.col('latitude').mean().alias('mean_lat'),
        pl.col('longitude').mean().alias('mean_lon')
    ).to_numpy().flatten()  # 1次元配列に変換
    
    # 中央区からのポテンシャル（距離の逆二乗に比例）を計算
    df = df.with_columns(
        (1 / ((pl.col('latitude') - Chuou_lat)**2 + (pl.col('longitude') - Chuou_lon)**2)).alias('Chuou_ward_Potential')
    )

    return df

def pipe_line(df_train: pl.DataFrame, df_test: pl.DataFrame, mode='train'):
    """
    訓練データとテストデータに一連の前処理を適用するパイプライン関数
    
    入力:
        df_train (pl.DataFrame): 訓練データ
        df_test (pl.DataFrame): テストデータ
        mode (str): 実行モード（デフォルト: 'train'）
    
    処理:
        1. 訓練データとテストデータの結合
        2. 各列に対する前処理の適用
           - 列名の英訳
           - 誤字脱字の修正
           - 各種特徴量エンジニアリング
        3. 共通処理の適用
           - テストデータの目的変数の補完
           - 新しい特徴量の生成
           - 異常値の削除
        4. カテゴリ変数の処理
    
    出力:
        tuple: (前処理済み訓練データ, 前処理済みテストデータ, カテゴリ列リスト)
    
    注意:
        このパイプライン関数のみをノートブックで使用する
    """
    # 訓練データとテストデータの結合
    df_train = df_train.with_columns(pl.lit(1).alias('is_train'))
    df_test = df_test.with_columns(
        pl.lit(None).alias('賃料'),          # テストデータの目的変数はNull
        pl.lit(0).alias('is_train')          # テストデータのフラグは0
    )
    df_test = df_test.select(df_train.columns) # 列を揃える
    df = pl.concat([df_train, df_test])       # データの結合
    
    # 列ごとの前処理適用
    df = rename_columns(df)                   # 列名の英訳
    df = modify_miss(df)                      # 誤字脱字の修正
    df = process_Location(df)                 # 所在地情報の処理
    df = process_Access(df)                   # アクセス情報の処理
    df = process_Floor_Plan(df)               # 間取り情報の処理
    df = process_Age_of_Building(df)          # 築年数情報の処理
    df = process_Area(df)                     # 面積情報の処理
    df = process_Story_and_Floor(df)          # 階数情報の処理
    df = process_Bath_and_Toilet(df)          # バス・トイレ情報の処理
    df = process_Broadcasting_and_Communication(df) # 放送・通信情報の処理
    df = process_Kitchen(df)                  # キッチン情報の処理
    df = process_Indoor_Facilities(df)        # 室内設備情報の処理
    df = process_Parking(df)                  # 駐車場情報の処理
    df = process_Surrounding_Environment(df)  # 周辺環境情報の処理
    df = process_Architecture(df)             # 建物構造情報の処理
    df = process_Contract_Period(df)          # 契約期間情報の処理
    
    # 列単位の前処理後の共通処理
    df = fill_in_target(df)                   # テストデータの目的変数を補完
    df = ensemble_columns(df)                 # 新しい特徴量の生成
    df = delete_outlier(df)                   # 異常値の削除
    
    # 不要な列の削除
    drop_list = ['Location']
    df = df.drop(drop_list)
    
    # Pandasに変換
    df = df.to_pandas()
    
    # カテゴリ変数の処理
    cat_cols = ['district', 'Direction', 'nearest_line_name']
    
    # 欠損値の処理とカテゴリ型への変換
    for cat_col in cat_cols:
        df[cat_col] = df[cat_col].fillna('missing').astype(str)
    for cat_col in cat_cols:
        df[cat_col] = df[cat_col].astype('category')
    
    # 訓練データとテストデータの分割
    df_train = df[df.Target.notna()].drop('is_train', axis=1)  # 目的変数が存在するデータ
    df_test = df[df.is_train == 0].drop('is_train', axis=1)    # テストデータフラグが0のデータ
    
    return df_train, df_test, cat_cols