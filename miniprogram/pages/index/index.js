const app = getApp()
const MAX_HISTORY = 10

Page({
  data: {
    imgSrc: '',
    isAnalyzing: false,
    result: null,
    error: '',
    history: [],
    statusBarHeight: 0,
    confidenceText: '0.0',
    confidenceWidth: '0%'
  },

  onLoad: function() {
    this.loadHistory()
    
    // 获取系统信息
    wx.getSystemInfo({
      success: (res) => {
        this.setData({
          statusBarHeight: res.statusBarHeight
        })
      }
    })
  },

  onShow: function() {
    // 每次页面显示时重新加载历史记录
    this.loadHistory()
  },

  loadHistory: function() {
    const history = wx.getStorageSync('history') || []
    this.setData({ history })
  },

  saveHistory: function(record) {
    let history = wx.getStorageSync('history') || []
    
    // 添加新记录到开头
    history.unshift({
      ...record,
      time: new Date().toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      })
    })
    
    // 限制历史记录数量
    if (history.length > MAX_HISTORY) {
      history = history.slice(0, MAX_HISTORY)
    }
    
    // 保存到本地存储
    wx.setStorageSync('history', history)
    
    // 更新页面数据
    this.setData({ history })
  },

  // 选择图片
  chooseImage() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album'],
      success: (res) => {
        const tempFilePath = res.tempFiles[0].tempFilePath;
        this.setData({
          imgSrc: tempFilePath,
          result: null,
          error: ''
        });
      }
    });
  },

  // 拍照
  takePhoto() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['camera'],
      success: (res) => {
        const tempFilePath = res.tempFiles[0].tempFilePath;
        this.setData({
          imgSrc: tempFilePath,
          result: null,
          error: ''
        });
        // 自动开始分析
        this.analyzeImage();
      }
    });
  },

  // 清除错误
  clearError() {
    this.setData({
      error: ''
    });
  },

  // 格式化结果数据
  formatResult(rawResult) {
    if (!rawResult) return null;
    
    return {
      ...rawResult,
      confidenceText: (rawResult.confidence * 100).toFixed(2) + '%',
      inferenceTimeText: rawResult.inference_time ? rawResult.inference_time.toFixed(3) + '秒' : '0.000秒',
      formattedProbabilities: Object.entries(rawResult.class_probabilities || {}).map(([key, value]) => ({
        name: key,
        percent: Number(value) * 100,
        text: (Number(value) * 100).toFixed(2) + '%'
      }))
    };
  },

  // 分析图片
  analyzeImage() {
    if (!this.data.imgSrc) {
      this.setData({
        error: '请先选择一张图片'
      });
      return;
    }

    this.setData({
      isAnalyzing: true,
      error: ''
    });

    // 将图片转为base64
    wx.getFileSystemManager().readFile({
      filePath: this.data.imgSrc,
      encoding: 'base64',
      success: (res) => {
        // 调用后端API
        wx.request({
          url: 'http://172.20.10.3:5000/predict',  
          method: 'POST',
          header: {
            'content-type': 'application/json'  // 确保设置正确的内容类型
          },
          data: {
            image: res.data
          },
          success: (response) => {
            console.log('请求成功', response);
            if (response.statusCode === 200 && response.data) {
              if (response.data.success) {
                const result = {
                  predicted_class: response.data.disease,
                  confidence: response.data.confidence,
                  inference_time: 0,
                  class_probabilities: response.data.class_probabilities
                };
                const formattedResult = this.formatResult(result);
                
                // 更新置信度显示
                this.setData({
                  result: formattedResult,
                  isAnalyzing: false,
                  confidenceText: (response.data.confidence * 100).toFixed(1),
                  confidenceWidth: (response.data.confidence * 100) + '%'
                });
                
                // 保存到历史记录
                this.saveHistory({
                  imagePath: this.data.imgSrc,
                  disease: response.data.disease,
                  confidence: response.data.confidence * 100
                });
              } else {
                this.setData({
                  error: response.data.error || '识别失败',
                  isAnalyzing: false
                });
              }
            } else {
              this.setData({
                error: '服务器响应异常: ' + response.statusCode,
                isAnalyzing: false
              });
            }
          },
          fail: (err) => {
            console.error('请求失败详情:', err);
            let errorMsg = '网络请求失败';
            
            if (err.errMsg.includes('url not in domain list')) {
              errorMsg = '请在微信开发者工具中开启"不校验合法域名"选项';
            }
            
            this.setData({
              error: errorMsg,
              isAnalyzing: false
            });
          }
        });
      },
      fail: (err) => {
        this.setData({
          error: '读取图片失败: ' + err.errMsg,
          isAnalyzing: false
        });
      }
    });
  },

  // 查看历史记录
  viewHistory(e) {
    const index = e.currentTarget.dataset.index;
    const item = this.data.history[index];
    if (item) {
      const formattedResult = this.formatResult({
        predicted_class: item.disease,
        confidence: item.confidence / 100,
        inference_time: 0
      });
      this.setData({
        imgSrc: item.imagePath,
        result: formattedResult,
        confidenceText: item.confidence.toFixed(1),
        confidenceWidth: item.confidence + '%'
      });
      
      // 滚动到顶部
      wx.pageScrollTo({
        scrollTop: 0,
        duration: 300
      });
    }
  },
  
  goBack: function() {
    this.setData({
      imgSrc: '',
      result: null,
      confidenceText: '0.0',
      confidenceWidth: '0%'
    });
  }
});
