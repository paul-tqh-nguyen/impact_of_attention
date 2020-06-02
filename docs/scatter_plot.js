
const data_location = "../visualization_data/accuracy_vs_number_of_parameters.json";

const svg = d3.select('#scatter-plot-svg');
const scatterPlotGroup = svg.append('g');
const attentionScatterPoints = scatterPlotGroup.append('g');
const plainRNNScatterPoints = scatterPlotGroup.append('g');
const scatterPlotTitle = scatterPlotGroup.append('text');
const xAxisGroup = scatterPlotGroup.append('g');
const xAxisLabel = xAxisGroup.append('text');
const yAxisGroup = scatterPlotGroup.append('g');
const yAxisLabel = yAxisGroup.append('text');
    
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
const attention_fill = "red";
const plain_rnn_fill = "blue";

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

    scatterPlotTitle
        .style('font-size', Math.min(20, innerWidth/40))
        .text("Test Accuracy vs Model Parameter Count")
        .attr('x', innerWidth * 0.225)
        .attr('y', -10);
    
    const yAxisTickFormat = number => d3.format('.3f')(number);
    yAxisGroup.call(d3.axisLeft(yScale).tickFormat(yAxisTickFormat).tickSize(-innerWidth));
    yAxisGroup.selectAll('.tick line')
        .style('opacity', innerLineOpacity);
    yAxisGroup.selectAll('.tick text')
        .attr('transform', 'translate(-3.0, 0.0)');
    yAxisLabel
        .style('font-size', 15)
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
        .style('font-size', 15)
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
        .attr('fill', attention_fill)
        .attr('fill-opacity', scatterPointFillOpacity);
    
    plainRNNScatterPoints.selectAll('circle').data(plain_rnn_data)
        .remove();
    plainRNNScatterPoints.selectAll('circle').data(plain_rnn_data)
        .enter()
        .append('circle')
        .attr('cy', datum => yScale(getDatumLoss(datum)))
        .attr('cx', datum => xScale(getDatumParameterCount(datum)))
        .attr('r', scatterPointRadius)
        .attr('fill', plain_rnn_fill)
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
