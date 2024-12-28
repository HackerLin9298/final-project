import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from configparser import ConfigParser

class BitcoinQA:
    def __init__(self, scaler_path, model_path, data_path, cleaned_data_file, gemini_api_key):
        self.scaler_path = scaler_path
        self.model_path = model_path
        self.data_path = data_path
        self.cleaned_data_file = cleaned_data_file
        self.scaler = None
        self.model = None

        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=gemini_api_key)
        prompt = ChatPromptTemplate.from_template(
            """請只用提供的context來回答以下的問題，如果無法回答，請說「此問題與本文無關」。
        <context>
        {context}
        </context>
        問題 : {input}。
        使用繁體中文回答問題。
        """
        )
        self.document_chain = create_stuff_documents_chain(self.llm, prompt)

        # Initialize FAISS database
        loader = TextLoader(self.cleaned_data_file, autodetect_encoding=True)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = FAISS.from_documents(docs, embeddings)

    def load_model_and_scaler(self):
        # Load scaler
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Load LSTM model
        self.model = load_model(self.model_path)

    def predict_next_week(self):
        # 加载清理后的数据
        data = pd.read_csv(self.data_path, sep='\t')

        # 提取特征
        features = data[['Open', 'High', 'Low', 'Vol.', 'MA7', 'MA21', 'EMA', 'High-Low', 'Price-Open']].values
        features_scaled = self.scaler.transform(features)

        # 准备最后一个数据点
        last_data_point = features_scaled[-1].reshape(1, -1, 1)

        # 预测未来 7 天
        predictions = []
        for _ in range(7):
            # 使用模型预测
            prediction_scaled = self.model.predict(last_data_point)  # 模型预测，输出形状为 (1, 1)

            # 单独对目标值进行逆缩放
            prediction = self.scaler.data_min_[-1] + (self.scaler.data_max_[-1] - self.scaler.data_min_[-1]) * prediction_scaled[0][0]

            predictions.append(prediction)  # 存储预测结果

            # 更新滑动窗口数据点
            last_data_point = np.roll(last_data_point, -1, axis=1)
            last_data_point[0, -1, 0] = prediction_scaled[0][0]  # 更新目标值

        return predictions



    def query(self, question):
        # Check if the question is about future price prediction
        if "未來一週" in question:
            future_predictions = self.predict_next_week()
            return f"未來一週的比特幣價格走勢預測是：{', '.join([f'{p:.2f}' for p in future_predictions])}。"

        # Otherwise, use Gemini for general questions
        results = self.db.similarity_search_with_score(question, 1)

        # 检查结果是否非空
        if results and len(results) > 0:
            # 提取第一个结果
            document = results[0][0]

            # 简化上下文内容
            if isinstance(document, str):
                raw_context = document
            elif hasattr(document, "page_content"):
                raw_context = document.page_content
            else:
                raise ValueError(f"Unsupported document type: {type(document)}")

            # 格式化上下文为自然语言描述
            simplified_context = self.format_context(raw_context)

            # 打印调试信息
            print(f"Formatted Context being passed to LLM: {simplified_context}")

            # 构建上下文字典
            context = {"page_content": simplified_context}

            # 调用 LLM 生成回答
            try:
                llm_result = self.document_chain.invoke(
                    {
                        "input": question,
                        "context": context,
                    }
                )
                return llm_result
            except Exception as e:
                return f"生成回答時出現錯誤：{e}"

        return "此問題與本文無關。"

    def format_context(self, raw_context):
        """
        将表格或复杂结构的内容转换为自然语言描述
        """
        try:
            # 将表格数据按行解析为自然语言
            lines = raw_context.split("\n")
            headers = lines[0].split("\t")
            data_rows = lines[1:]

            # 构建自然语言描述
            descriptions = []
            for row in data_rows[:3]:  # 仅取前 3 行示例
                values = row.split("\t")
                entry = {headers[i]: values[i] for i in range(len(headers))}
                descriptions.append(
                    f"日期為 {entry['Date']}，收盤價為 {entry['Price']} 美元，開盤價為 {entry['Open']} 美元，最高價為 {entry['High']} 美元，最低價為 {entry['Low']} 美元，交易量為 {entry['Vol.']}，當日漲跌幅為 {entry['Change %']}%。"
                )

            # 将描述合并为单段文本
            return " ".join(descriptions)

        except Exception as e:
            return f"上下文格式化時出現錯誤：{e}"



if __name__ == "__main__":
    # Load configuration
    config = ConfigParser()
    config.read("config.ini")

    # Paths to required files
    scaler_path = "/Users/linyoucheng/ml/Final project/scaler.pkl"

    model_path = "lstm_model.h5"
    data_path = "bitcoin_cleaned.txt"
    cleaned_data_file = "bitcoin_cleaned.txt"

    # Initialize QA system
    qa_system = BitcoinQA(scaler_path, model_path, data_path, cleaned_data_file, config["Gemini"]["API_KEY"])
    qa_system.load_model_and_scaler()

    # Example queries
    print(qa_system.query("未來一週的價格走勢是什麼？"))
    print(qa_system.query("未來3天的價格走勢是什麼？"))
