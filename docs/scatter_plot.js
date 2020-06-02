
const data_location = "./visualization_data/accuracy_vs_number_of_parameters.json";

const svg = d3.select('#scatter-plot-svg');
const scatterPlotGroup = svg.append('g');
const attentionScatterPoints = scatterPlotGroup.append('g');
const plainRNNScatterPoints = scatterPlotGroup.append('g');
const scatterPlotTitle = scatterPlotGroup.append('text');
const xAxisGroup = scatterPlotGroup.append('g');
const xAxisLabel = xAxisGroup.append('text');
const yAxisGroup = scatterPlotGroup.append('g');
const yAxisLabel = yAxisGroup.append('text');
const legend = scatterPlotGroup.append('g');
const legendBoundingBox = legend.append('rect');
const attentionLegendText = legend.append('text');
const plainRNNLegendText = legend.append('text');
const attentionLegendCircle = legend.append('circle');
const plainRNNLegendCircle = legend.append('circle');

const getDatumLoss = datum => datum.test_loss;
const getDatumParameterCount = datum => datum.number_of_parameters;

const margin = {
    top: 80,
    bottom: 80,
    left: 120,
    right: 30,
};

const innerLineOpacity = 0.1;
const xAxisRightPaddingAmount = 1000000;

const scatterPointRadius = 3;
const scatterPointFillOpacity = 0.3;
const attentionFill = "red";
const plainRNNFill = "blue";
const attentionLegendKeyText = "Attention";
const plainRNNLegendKeyText = "Plain RNN";

const render = (attention_data, plain_rnn_data) => {
    
    const plotContainer = document.getElementById("scatter-plot");
    svg
        .attr('width', plotContainer.clientWidth)
        .attr('height', plotContainer.clientHeight);

    const svg_width = parseFloat(svg.attr('width'));
    const svg_height = parseFloat(svg.attr('height'));
    
    const innerWidth = svg_width - margin.left - margin.right;
    const innerHeight = svg_height - margin.top - margin.bottom;
    
    const xMaxValue = Math.max(d3.max(attention_data, getDatumParameterCount), d3.max(plain_rnn_data, getDatumParameterCount));
    const xScale = d3.scaleLinear()
          .domain([0, xMaxValue+xAxisRightPaddingAmount])
          .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
          .domain([1.0, 0.0])
          .range([0, innerHeight]);
    
    scatterPlotGroup.attr('transform', `translate(${margin.left}, ${margin.top})`);

    const labelSize = Math.min(20, innerWidth/40)
    scatterPlotTitle
        .style('font-size', labelSize)
        .text("Test Accuracy vs Model Parameter Count")
        .attr('x', innerWidth * 0.225)
        .attr('y', -10);

    const legendKeyFontSize = Math.min(15, innerWidth/40);
    const attentionLegendKeyX = innerWidth - attentionLegendKeyText.length * legendKeyFontSize;
    const attentionLegendKeyY = innerHeight - legendKeyFontSize * 4.5;
    const plainRNNLegendKeyX = innerWidth - plainRNNLegendKeyText.length * legendKeyFontSize;
    const plainRNNLegendKeyY = innerHeight - legendKeyFontSize * 3;
    const legendBoundingBoxX = attentionLegendKeyX - legendKeyFontSize / 2;
    const legendBoundingBoxY = attentionLegendKeyY - legendKeyFontSize * 1.5;
    const legendBoundingBoxWidth = Math.max(attentionLegendKeyText.length, plainRNNLegendKeyText.length) * legendKeyFontSize * 0.75;
    const legendBoundingBoxHeight = legendKeyFontSize * 4;
    legendBoundingBox
        .attr('x', legendBoundingBoxX)
        .attr('y', legendBoundingBoxY)
        .attr('width', legendBoundingBoxWidth)
        .attr('height', legendBoundingBoxHeight)
        .style('stroke-width', 1)
        .style('stroke', 'black')
        .attr('fill', 'white');
    attentionLegendCircle
        .attr('cx', attentionLegendKeyX + legendKeyFontSize / 2)
        .attr('cy', attentionLegendKeyY - legendKeyFontSize * 0.75 + legendKeyFontSize / 2)
        .attr('r', legendKeyFontSize / 2)
        .attr('fill', attentionFill);
    plainRNNLegendCircle
        .attr('cx', plainRNNLegendKeyX + legendKeyFontSize / 2)
        .attr('cy', plainRNNLegendKeyY - legendKeyFontSize * 0.75 + legendKeyFontSize / 2)
        .attr('r', legendKeyFontSize / 2)
        .attr('fill', plainRNNFill);
    attentionLegendText
        .style('font-size', legendKeyFontSize)
        .html(attentionLegendKeyText)
        .attr('x', attentionLegendKeyX + legendKeyFontSize * 1.25)
        .attr('y', attentionLegendKeyY)
        .attr('stroke', attentionFill);
    plainRNNLegendText
        .style('font-size', legendKeyFontSize)
        .html(plainRNNLegendKeyText)
        .attr('x', plainRNNLegendKeyX + legendKeyFontSize * 1.25)
        .attr('y', plainRNNLegendKeyY)
        .attr('stroke', plainRNNFill);
    
    const yAxisTickFormat = number => d3.format('.3f')(number);
    yAxisGroup.call(d3.axisLeft(yScale).tickFormat(yAxisTickFormat).tickSize(-innerWidth));
    yAxisGroup.selectAll('.tick line')
        .style('opacity', innerLineOpacity);
    yAxisGroup.selectAll('.tick text')
        .attr('transform', 'translate(-3.0, 0.0)');
    yAxisLabel
        .style('font-size', labelSize * 0.8)
        .attr('fill', 'black')
        .attr("transform", "rotate(-90)")
        .attr('y', -60)
        .attr('x', -innerHeight/3)
        .text('Mean Cross Entropy Loss');
    
    const xAxisTickFormat = number => d3.format('.3s')(number).replace(/G/,"B");
    xAxisGroup.call(d3.axisBottom(xScale).tickFormat(xAxisTickFormat).tickSize(-innerHeight))
        .attr('transform', `translate(0, ${innerHeight})`);
    xAxisGroup.selectAll('.tick line')
        .style('opacity', innerLineOpacity);
    xAxisGroup.selectAll('.tick text')
        .attr('transform', 'translate(0.0, 5.0)');
    xAxisLabel
        .style('font-size', labelSize * 0.8)
        .attr('fill', 'black')
        .attr('y', margin.bottom * 0.75)
        .attr('x', innerWidth / 2)
        .text('Parameter Count');

    attentionScatterPoints.selectAll('circle').data(attention_data)
        .remove();
    attentionScatterPoints.selectAll('circle').data(attention_data)
        .enter()
        .append('circle')
        .attr('cy', datum => yScale(getDatumLoss(datum)))
        .attr('cx', datum => xScale(getDatumParameterCount(datum)))
        .attr('r', scatterPointRadius)
        .attr('fill', attentionFill)
        .attr('fill-opacity', scatterPointFillOpacity);
    
    plainRNNScatterPoints.selectAll('circle').data(plain_rnn_data)
        .remove();
    plainRNNScatterPoints.selectAll('circle').data(plain_rnn_data)
        .enter()
        .append('circle')
        .attr('cy', datum => yScale(getDatumLoss(datum)))
        .attr('cx', datum => xScale(getDatumParameterCount(datum)))
        .attr('r', scatterPointRadius)
        .attr('fill', plainRNNFill)
        .attr('fill-opacity', scatterPointFillOpacity);

};

const redraw = () => {
    d3.json(data_location)
        .then(data => {
            let extract_data = datum => {
                return {
                    test_loss: parseFloat(datum.test_loss),
                    test_accuracy: parseFloat(datum.test_accuracy),
                    number_of_parameters: parseInt(datum.number_of_parameters)
                };
            };
            let attention_data = data['attention'].map(extract_data);
            let plain_rnn_data = data['plain_rnn'].map(extract_data);
            render(attention_data, plain_rnn_data);
        }).catch(err => {
            console.error(err.message);
            return;
        });
};

redraw();
window.addEventListener("resize", redraw);
