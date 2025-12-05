# 🚛 TSP Optimization - Tối ưu hóa lộ trình giao hàng

Giải bài toán **Travelling Salesman Problem (TSP)** bằng **Genetic Algorithm (GA)** và **Particle Swarm Optimization (PSO)** - Ứng dụng cho tối ưu lộ trình giao hàng.

## 📋 Tổng quan dự án

### Đề tài
"Giải bài toán Người du lịch (TSP) bằng Giải thuật Di truyền (GA) và thuật toán từ nhóm Tối ưu Hóa Bầy Đàn (PSO) - Ứng dụng cho tối ưu lộ trình giao hàng"

### Yêu cầu
- ✅ **Bắt buộc**: Sử dụng Genetic Algorithm (GA) 
- ✅ **Bắt buộc**: Sử dụng Particle Swarm Optimization (PSO)
- 🎯 **Mục tiêu**: So sánh hiệu quả của 2 thuật toán

### Bài toán thực tế
- **Input**: Tài xế có 1 vị trí xuất phát và N điểm giao hàng
- **Yêu cầu**: Tìm lộ trình đi qua tất cả các điểm với tổng quãng đường ngắn nhất
- **Output**: Thứ tự các điểm cần đi và tổng quãng đường tối ưu

## 🛠️ Tech Stack

- **Python**: 3.11.9
- **Algorithms**: DEAP 1.4.3 (GA), Custom PSO
- **Frontend**: Streamlit 1.28+
- **Maps**: Google Maps API
- **Data**: NumPy, Pandas
- **Visualization**: Matplotlib, Plotly, Seaborn

## 📁 Cấu trúc dự án

tsp-optimization/
├── src/ # Core algorithms
├── components/ # UI components
├── data/test_cases/ # Test data
├── tests/ # Unit tests
├── docs/ # Documentation
├── notebooks/ # Research notebooks
└── app.py # Main application

## 🚀 Installation & Setup

### 1. Clone repository
git clone: 
cd tsp-optimization
### 2. Setup Python environment
Tạo virtual environment (Python 3.11.9 required)
python -m venv venv
source venv/bin/activate # Linux/Mac
.venv\Scripts\activate # Windows
Upgrade pip
pip install --upgrade pip
### 3. Install dependencies
pip install -r requirements.txt
### 4. Setup environment
Copy environment template
cp .env.example .env
Edit .env file - add your Google Maps API key
GOOGLE_MAPS_API_KEY=your_api_key_here
### 5. Run application
streamlit run app.py

## 👥 Team Members & Responsibilities

- **Hoàng** (Team Leader):  
✅ Thiết kế kiến trúc tổng thể hệ thống
✅ Implement Genetic Algorithm (GA) - thuật toán chính 
✅ Xây dựng TSPSolver base class - nền tảng cho cả GA và PSO 
✅ Tích hợp toàn bộ hệ thống trong app.py 
✅ Quản lý configuration, dependencies, documentation
✅ Code review và merge code của team

- **Quang** (Algorithm Specialist): 
✅ Implement Particle Swarm Optimization (PSO) - thuật toán thứ 2 (7.1 KB)
✅ Xây dựng Algorithm Comparison Framework - so sánh hiệu năng GA vs PSO 
✅ Phát triển Visualizer module - biểu đồ convergence, performance charts 
✅ Testing và đánh giá hiệu năng các thuật toán
✅ Tối ưu hóa parameters cho PSO
✅ Benchmark và statistical analysis

- **Quân** (UI/UX Developer, Map Visualization Expert):   
✅ Đồng phát triển GeoUtils với Hoàng - phần visualization 
✅ Xây dựng Interactive Map UI - Folium integration, click handling 
✅ Thiết kế giao diện tương tác bản đồ
✅ Implement location picker, marker management
✅ Route visualization trên map
✅ OpenStreetMap integration

- **Nhân** (UI Developer, Results Display Specialist): 
✅ Đồng phát triển Visualizer với Quang - phần UI display (12.8 KB shared)
✅ Xây dựng Sidebar Configuration UI - parameter controls (6.7 KB)
✅ Phát triển Results Display Module - charts, tables, metrics (15.8 KB - file UI lớn nhất)
✅ Thiết kế data presentation layer
✅ Integration testing cho UI components
✅ Export functionality (CSV, JSON)


## 📊 Features

### Core Algorithms
- [x] Genetic Algorithm (GA) 
- [x] Particle Swarm Optimization (PSO)
- [x] Performance comparison framework

### UI Features  
- [x] Interactive OpenStreetMaps interface
- [x] Parameter configuration sidebar
- [x] Real-time algorithm visualization
- [x] Results comparison dashboard

### Data Features
- [x] OpenStreetMap Distance Matrix API integration
- [x] Intelligent caching system
- [x] Export results (CSV, JSON)
- [x] Test cases management

## 🧪 Testing
Run all tests
python -m pytest tests/

Run specific test
python tests/test_ga_solver.py

Check installation
python test_deap_setup.py

## 📈 Performance

Benchmarks on 20-city TSP instances:
- **GA**: Average 95% of optimal in 200 generations
- **PSO**: Average 92% of optimal in 100 iterations  
- **Runtime**: < 10 seconds on modern hardware

## 🤝 Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add feature'` 
4. Push branch: `git push origin feature/your-feature`
5. Submit pull request

## 📄 License

This project is developed for educational purposes at UEH University.

---
**Developed by Team**: Hoàng, Quang, Quân, Nhân | **University**: UEH | **Year**: 2025