// ... existing code ...

// 确保只将置信度乘以100一次
const confidencePercent = (result.confidence * 100).toFixed(2) + '%';
this.setData({
  confidenceText: confidencePercent
});

// ... existing code ...